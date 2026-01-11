import os
import time
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.markup import escape

from .utils import get_agno_model_class
from .key_rotation.rotation_manager import RotatingKeyManager
from .key_rotation.rotating_mixin import RotatingCredentialsMixin
from .config.dataclasses import KeyUsage, RateLimits, UsageSnapshot
from .config.enums import RateLimitStrategy
from .config.constants import MODEL_LIMITS, PROVIDER_STRATEGIES
from .usage.db_logic import UsageDatabase
from .config.log_config import default_logger
from .adapters import RotatingOpenAIClient, RotatingAsyncOpenAIClient

class MultiProviderWrapper:
    """Wrapper for Agno models with rotating API keys"""
    
    PROVIDER_STRATEGIES = PROVIDER_STRATEGIES
    MODEL_LIMITS = MODEL_LIMITS
    
    @staticmethod
    def load_api_keys(provider: str, env_file: Optional[str] = None) -> List[str]:
        """Load API keys from environment variables"""
        if env_file:
            env_path = Path(env_file).resolve()
            # if not env_path.exists():
                 # print(f"Warning: The provided env_file '{env_path}' does not exist.")
            #     logger.warning("The provided env_file '%s' does not exist.", env_path)
        else:
            env_path = Path.cwd() / ".env"
        load_dotenv(dotenv_path=env_path, override=True)
        num_keys_var = f"NUM_{provider.upper()}"
        num_keys = os.getenv(num_keys_var)
        if not num_keys:
            raise ValueError(f"Environment variable '{num_keys_var}' not found.")
        try:
            num_keys = int(num_keys)
        except ValueError:
            raise ValueError(f"'{num_keys_var}' must be an integer, got: {num_keys}")

        api_keys = []
        for i in range(1, num_keys + 1):
            key_var = f"{provider.upper()}_API_KEY_{i}"
            key = os.getenv(key_var)
            if not key: raise ValueError(f"Missing API key: {provider.upper()}_API_KEY_{i}")
            api_keys.append(key)
        return api_keys
    
    @classmethod
    def from_env(
        cls, 
        provider: str, 
        default_model_id: str, 
        env_file: Optional[str] = None,
        db_url: Optional[str] = None,
        db_env_var: Optional[str] = "TIDB_DB_URL",
        debug: bool = False, 
        logger = None,
        **kwargs
    ):
        try:
            model_class = get_agno_model_class(provider)
        except (ValueError, ImportError):
            # Agno might not be installed, or provider not supported by Agno utils.
            # We proceed without a model class, allowing only OpenAI client usage.
            model_class = None

        api_keys = cls.load_api_keys(provider, env_file)
        db_url = db_url or os.getenv(db_env_var)
        return cls(provider, api_keys, default_model_id, 
                   model_class, db_url, db_env_var, debug, logger, **kwargs)
    
    def __init__(
        self, 
        provider: str, 
        api_keys: List[str], 
        default_model_id: str,
        model_class: Optional[Any] = None, 
        db_url: Optional[str] = None, 
        db_env_var: Optional[str] = "TIDB_DB_URL",
        debug: bool = False, logger = None,
        **kwargs
    ):
        self.provider = provider.lower()
        self.logger = logger or default_logger
        self.model_class = model_class
        self.default_model_id = default_model_id
        self.model_kwargs = kwargs
        self.toggle_debug(debug)
        self.db = UsageDatabase(db_url, db_env_var)
        self.strategy = self.PROVIDER_STRATEGIES.get(self.provider, RateLimitStrategy.PER_MODEL)
        self.manager = RotatingKeyManager(api_keys, self.provider, self.strategy, self.db)
        self._model_cache = {}
        self._RotatingClass = None
        self.console = Console()

    def toggle_debug(self, enable: bool):
        """
        Dynamically switches logging verbosity for this module.
        enable=True  -> Shows detailed rotation/reservation logs (DEBUG)
        enable=False -> Shows only key info/warnings (INFO)
        """
        level = logging.DEBUG if enable else logging.INFO
        # Set the logger for the current file context
        self.logger.setLevel(level)
        
        status = "ENABLED" if enable else "DISABLED"
        self.logger.info(f"Debug logging {status} for {self.provider}")

    def _resolve_limits(self, model_id: str) -> RateLimits:
        mid = model_id or self.default_model_id
        provider_limits = self.MODEL_LIMITS.get(self.provider, {})
        return provider_limits.get(mid, provider_limits.get('default', RateLimits(10, 100, 1000)))

    def get_key_usage(
        self, 
        model_id: str = None, 
        estimated_tokens: int = 1000, 
        wait: bool = True, 
        timeout: float = 10
    ):
        """Finds the first valid key"""
        mid = model_id or self.default_model_id
        limits = self._resolve_limits(mid)

        start = time.time()
        while True:
            key_usage = self.manager.get_key(mid, limits, estimated_tokens)
            if key_usage: 
                return key_usage
            if not wait:
                    raise RuntimeError(f"No available API keys for {self.provider}/{mid} (wait=False)") 
            if time.time() - start > timeout:
                raise RuntimeError(f"Timeout: No available API keys for {self.provider}/{mid} after {timeout}s")
            time.sleep(0.5)

    def get_openai_client(
        self, 
        estimated_tokens: int = 1000, 
        max_retries: int = 5, 
        **kwargs
    ) -> RotatingOpenAIClient:
        """
        Returns a rotating OpenAI client (Sync)
        
        Args:
            estimated_tokens: Estimated tokens per request for rate limiting
            max_retries: Maximum retries on rate limit errors
            **kwargs: Additional arguments passed to the OpenAI client
        """
        return RotatingOpenAIClient(
            manager=self.manager,
            limit_resolver=self._resolve_limits,
            default_model=self.default_model_id,
            estimated_tokens=estimated_tokens,
            max_retries=max_retries,
            provider=self.provider,  # Pass provider so it can look up base_url
            client_kwargs={**self.model_kwargs, **kwargs},
        )

    def get_async_openai_client(
        self, 
        estimated_tokens: int = 1000, 
        max_retries: int = 5, 
        **kwargs
    ) -> RotatingAsyncOpenAIClient:
        """
        Returns a rotating OpenAI client (Async)
        
        Args:
            estimated_tokens: Estimated tokens per request for rate limiting
            max_retries: Maximum retries on rate limit errors
            **kwargs: Additional arguments passed to the AsyncOpenAI client
        """
        return RotatingAsyncOpenAIClient(
            manager=self.manager,
            limit_resolver=self._resolve_limits,
            default_model=self.default_model_id,
            estimated_tokens=estimated_tokens,
            max_retries=max_retries,
            provider=self.provider,  # Pass provider so it can look up base_url
            client_kwargs={**self.model_kwargs, **kwargs}
        )

    def get_model(
        self, 
        estimated_tokens: int = 1000, 
        wait: bool = True, 
        timeout: float = 10, 
        max_retries: int = 5, 
        **kwargs
    ):
        """Dynamically creates a rotating model for ANY provider."""
        if self.model_class is None:
            raise RuntimeError(
                "Agno model class is not available. "
                "Ensure 'agno' is installed and that the provider is supported."
            )

        if self._RotatingClass is None:
            self._RotatingClass = type(
                f"Rotating{self.model_class.__name__}",
                (RotatingCredentialsMixin, self.model_class),
                {}
            )
        RotatingProviderClass = self._RotatingClass
        # Get Initial Key
        model_id = kwargs.get('id', self.default_model_id)  
        final_kwargs = {**self.model_kwargs, **kwargs}
        if 'id' not in final_kwargs:
            final_kwargs['id'] = model_id

        initial_key_usage = self.get_key_usage(model_id, estimated_tokens, wait=wait, timeout=timeout)
        

        model_instance = RotatingProviderClass(
            api_key=initial_key_usage.api_key,
            model_id = model_id,
            wrapper=self,
            rotating_wait=wait,
            rotating_timeout=timeout,
            rotating_estimated_tokens=estimated_tokens,
            rotating_max_retries=max_retries,
            **final_kwargs
        )

        # orig_metrics = getattr(model_instance, "_get_metrics", None)
        # def metrics_hook(*args, **kwargs):
        #     if orig_metrics:
        #         m = orig_metrics(*args, **kwargs)
        #     else:
        #         m = None
        #     actual = 0
        #     if m and hasattr(m, 'total_tokens') and m.total_tokens is not None:
        #         actual = m.total_tokens
        #     # Retrieve the estimate we set on the instance earlier
        #     estimate = getattr(model_instance, "_estimated_tokens", 1000)
        #     self.manager.record_usage(
        #         key_obj=initial_key_usage,
        #         model_id=model_id, 
        #         actual_tokens=actual, 
        #         estimated_tokens=estimate
        #     )
        #     return m
        
        # model_instance._get_metrics = metrics_hook
        return model_instance

    def get_api_key(
        self,
        model_id: Optional[str] = None,
        estimated_tokens: int = 1000,
        wait: bool = True,
        timeout: float = 10
    ) -> str:
        """
        Get a valid API key for direct use (e.g., embeddings, custom endpoints).
        
        Args:
            model_id: Model identifier for rate limiting (uses default if None)
            estimated_tokens: Estimated tokens for this request
            wait: Whether to wait for an available key
            timeout: Maximum time to wait for a key
            
        Returns:
            A valid API key string
            
        Example:
            >>> wrapper = MultiProviderWrapper.from_env('cohere', 'command-r-plus')
            >>> api_key = wrapper.get_api_key()
            >>> # Use with cohere SDK directly
            >>> import cohere
            >>> co = cohere.Client(api_key)
            >>> response = co.embed(texts=["hello"], model="embed-english-v3.0")
        """
        key_usage = self.get_key_usage(model_id, estimated_tokens, wait, timeout)
        return key_usage.api_key

    def get_api_key_with_context(
        self,
        model_id: Optional[str] = None,
        estimated_tokens: int = 1000,
        wait: bool = True,
        timeout: float = 10
    ) -> tuple[str, KeyUsage]:
        """
        Get an API key along with its usage context for manual tracking.
        This gives you both the key and the key_usage object for more control.
        
        Args:
            model_id: Model identifier for rate limiting
            estimated_tokens: Estimated tokens for this request
            wait: Whether to wait for an available key
            timeout: Maximum time to wait
            
        Returns:
            Tuple of (api_key: str, key_usage_obj: KeyUsage)
        """
        key_usage = self.get_key_usage(model_id, estimated_tokens, wait, timeout)
        return key_usage.api_key, key_usage

    def record_key_usage(
        self,
        api_key: str,
        model_id: Optional[str] = None,
        actual_tokens: int = 0,
        estimated_tokens: int = 1000
    ):
        """
        Record usage for a key obtained via get_api_key().
        Call this after you're done using the key to update usage tracking.
        
        Args:
            api_key: The API key that was used
            model_id: Model that was used (uses default if None)
            actual_tokens: Actual tokens consumed (if known)
            estimated_tokens: Estimated tokens (used if actual unknown)
            
        Example:
            >>> api_key = wrapper.get_api_key(estimated_tokens=500)
            >>> # ... use api_key for embeddings ...
            >>> wrapper.record_key_usage(api_key, model_id="embed-english-v3.0", actual_tokens=450)
        """
        mid = model_id or self.default_model_id
        
        # Find the key_usage object for this api_key
        key_obj = next((k for k in self.manager.keys if k.api_key == api_key), None)
        if key_obj:
            self.manager.record_usage(key_obj, mid, actual_tokens, estimated_tokens)
        else:
            self.logger.warning("API key not found in manager for recording usage")

    # --- PRINTING HELPERS ---
    
    def _create_usage_table(self, title: str, data: List[tuple[str, UsageSnapshot]]) -> Table:
        """
        Generates a standardized table for usage stats.
        data format: [(Label, Snapshot), ...]
        """
        # Palette
        c_title = "#bae1ff"  # Pastel Rose
        c_head  = "#f2f2f2"  # Pastel Cream
        c_req   = "#faa0a0"  # Pastel Periwinkle
        c_tok   = "#e5baff"  # Pastel Peach
        c_border= "#B9B9B9"  # Muted Grey
        c_identifier = "#7cd292"  # Soft Mauve
        
        table = Table(
            title=title, 
            box=box.ROUNDED, 
            expand=False, 
            title_style=f"bold {c_title}",
            title_justify="left",
            border_style=c_border,
            header_style=f"{c_head}"
        )

        # Define Columns
        table.add_column("Identifier", style=f"bold {c_identifier}", no_wrap=True)
        table.add_column("Requests ([white]m / h / d[/])",  justify="center", style=c_req, no_wrap=True)
        table.add_column("Tokens ([white]m / h / d[/])",  justify="center", style=c_tok, no_wrap=True)
        table.add_column("Total Requests", justify="center", style=f"{c_req}", no_wrap=True)
        table.add_column("Total Tokens", justify="center", style=f"bold {c_tok}", no_wrap=True)

        for label, s in data:
            req_str = f"{s.rpm} / {s.rph} / {s.rpd}"
            tok_str = f"{s.tpm:,} / {s.tph:,} / {s.tpd:,}"
            
            table.add_row(
                label,
                req_str,
                tok_str,
                f"{s.total_requests}",
                f"{s.total_tokens:,}"
            )
        return table

    def print_global_stats(self):
        stats = self.manager.get_global_stats()
        
        # 1. Prepare Data for the Table
        rows = []
        for k in stats.keys:
            label = f"Key #{k.index+1} (..{k.suffix})"
            rows.append((label, k.snapshot))
            
        # 2. Create and Print Table
        table = self._create_usage_table(
            title=f"GLOBAL STATS: {escape(self.provider.upper())}", 
            data=rows
        )
        
        # 3. Add a Summary Footer (using a Panel for the Total)
        total_s = stats.total
        grid = Table.grid(padding=(0, 4)) 
        grid.add_column(style="#e0e0e0") # Label Color
        grid.add_column(style="bold", justify="left") # Value Color

        grid.add_row("Total Requests:", f"[{'#faa0a0'}]{total_s.total_requests}[/]")
        grid.add_row("Total Tokens:",   f"[{'#e5baff'}]{total_s.total_tokens:,}[/]")
        
        self.console.print()
        self.console.print(Panel(
            grid, 
            title="[bold #bae1ff] AGGREGATE TOTALS [/]", 
            border_style="#bae1ff",
            expand=False
        ))
        self.console.print(table)

    def print_key_stats(self, identifier: Union[int, str]):
        stats = self.manager.get_key_stats(identifier)
        if not stats:
            self.console.print(f"[bold red]Key not found:[/][white] {identifier}[/]")
            return
        
        self.console.print()
        self.console.rule(f"[bold]Key Report: {stats.suffix}[/]")
        
        # 1. Total Snapshot Panel
        s = stats.total
        grid = Table.grid(padding=(0, 4))
        grid.add_column(style="#e0e0e0")
        grid.add_column(justify="left")

        grid.add_row("Total Requests:", f"[{'#faa0a0'}]{s.total_requests}[/]")
        grid.add_row("Total Tokens:",   f"[{'#e5baff'}]{s.total_tokens:,}[/]")
        
        self.console.print(Panel(
            grid, 
            title=f"[bold #97e3e9]Key #{stats.index+1} Overview[/]", 
            border_style="#bae1ff",
            expand=False
        ))

        # 2. Breakdown Table
        if not stats.breakdown:
            self.console.print("[italic dim]No usage recorded for this key yet.[/]")
        else:
            rows = [(model_id, snap) for model_id, snap in stats.breakdown.items()]
            table = self._create_usage_table(title="Breakdown by Model", data=rows)
            self.console.print(table)

    def print_model_stats(self, model_id: str):
        data = self.manager.get_model_stats(model_id)
        
        self.console.print()
        self.console.rule(self.console.rule(f"[bold]Model Report: [blue]{model_id}[/][/] ", style="#B9B9B9"))
        
        # 1. Total Summary
        s = data.total
        self.console.print(f"Total Tokens Consumed: [bold green]{s.total_tokens:,}[/]")
        
        # 2. Contributing Keys Table
        if not data.keys:
            self.console.print("[italic dim]No keys have used this model.[/]")
        else:
            rows = []
            for k in data.keys:
                label = f"Key #{k.index+1} (..{k.suffix})"
                rows.append((label, k.snapshot))
            
            table = self._create_usage_table(title="Contributing Keys", data=rows)
            self.console.print(table)

    def print_granular_stats(self, identifier: Union[int, str], model_id: str):
        data = self.manager.get_granular_stats(identifier, model_id)
        
        if not data:
            self.console.print(f"[bold red]Key '{identifier}' not found.[/]")
            return

        self.console.print()
        if data.snapshot:
            # Re-use the table builder for a single row just for consistency
            label = f"Key #{data.index+1} (..{data.suffix})"
            table = self._create_usage_table(
                title=f"Granular: {model_id}", 
                data=[(label, data.snapshot)]
            )
            self.console.print(table)
        else:
            self.console.print(Panel(
                f"No usage for model [bold]{model_id}[/] on key [bold]..{data.suffix}[/]",
                style="#e5baff",
                border_style="#B9B9B9"
            ))