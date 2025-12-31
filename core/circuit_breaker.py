"""
Circuit Breaker pattern implementation for fault tolerance.

Prevents cascade failures by monitoring failure rates and temporarily
disabling calls to failing services.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Dict, Optional

_logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascade failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, reject requests immediately
    - HALF_OPEN: Testing if service recovered, allow limited requests
    
    Configuration:
    - failure_threshold: Number of failures before opening circuit
    - recovery_timeout: Seconds to wait before trying again (half-open)
    - half_open_max_calls: Max calls allowed in half-open state
    
    Usage:
        circuit = get_circuit_breaker("my_service")
        if circuit.can_execute():
            try:
                result = call_service()
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure(e)
                raise
        else:
            # Service is unavailable, use fallback
            return fallback_value
    """
    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1
    
    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state (thread-safe)."""
        with self._lock:
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self.state == CircuitState.OPEN
    
    @property
    def failure_count(self) -> int:
        """Current failure count."""
        with self._lock:
            return self._failure_count
    
    @property
    def success_count(self) -> int:
        """Current success count since last failure."""
        with self._lock:
            return self._success_count
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try resetting the circuit."""
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def can_execute(self) -> bool:
        """
        Check if a request can be executed through the circuit.
        
        Returns True if:
        - Circuit is CLOSED (normal operation)
        - Circuit is OPEN but recovery timeout has passed (transitions to HALF_OPEN)
        - Circuit is HALF_OPEN and hasn't exceeded max test calls
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    _logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    return True
                return False
            
            # HALF_OPEN state
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    def record_success(self) -> None:
        """
        Record a successful operation.
        
        In HALF_OPEN state, this closes the circuit (recovery complete).
        In CLOSED state, this resets the failure counter.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                _logger.info(f"Circuit breaker '{self.name}' recovered, transitioning to CLOSED")
            self._failure_count = 0
            self._success_count += 1
    
    def record_failure(self, error: Optional[Exception] = None) -> None:
        """
        Record a failed operation.
        
        In HALF_OPEN state, this reopens the circuit immediately.
        In CLOSED state, this increments failure count and may open the circuit.
        """
        with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                _logger.warning(f"Circuit breaker '{self.name}' failed during recovery, reopening")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                _logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self._failure_count} failures"
                    f"{f': {error}' if error else ''}"
                )
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            _logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")
    
    def get_stats(self) -> Dict[str, any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }


# ---------------------------------------------------------------------------
# Global circuit breaker registry
# ---------------------------------------------------------------------------

# Default circuit breakers for different services
_CIRCUIT_BREAKERS: Dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 1,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    If a circuit breaker with the given name doesn't exist, creates one
    with the specified parameters. If it exists, returns the existing one
    (ignoring the parameters).
    
    Args:
        name: Unique identifier for the circuit breaker
        failure_threshold: Failures before opening (default: 5)
        recovery_timeout: Seconds before retrying (default: 30)
        half_open_max_calls: Test calls in half-open state (default: 1)
    
    Returns:
        CircuitBreaker instance
    """
    with _REGISTRY_LOCK:
        if name not in _CIRCUIT_BREAKERS:
            _CIRCUIT_BREAKERS[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
            )
        return _CIRCUIT_BREAKERS[name]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to closed state."""
    with _REGISTRY_LOCK:
        for cb in _CIRCUIT_BREAKERS.values():
            cb.reset()


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, any]]:
    """Get statistics for all circuit breakers."""
    with _REGISTRY_LOCK:
        return {name: cb.get_stats() for name, cb in _CIRCUIT_BREAKERS.items()}


# Pre-register common circuit breakers with appropriate defaults
def _init_default_breakers():
    """Initialize default circuit breakers for common services."""
    get_circuit_breaker("llm", failure_threshold=3, recovery_timeout=120.0)
    get_circuit_breaker("dense_retrieval", failure_threshold=3, recovery_timeout=60.0)
    get_circuit_breaker("bm25_retrieval", failure_threshold=5, recovery_timeout=30.0)
    get_circuit_breaker("rerank", failure_threshold=3, recovery_timeout=60.0)


_init_default_breakers()
