"""
Alert management — rule-based alerting on signal confidence drops,
model staleness, and portfolio drawdown.
"""
from datetime import datetime, timezone
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)

AlertHandler = Callable[[str, str, dict], None]


class Alert:
    def __init__(self, name: str, severity: str, message: str, context: dict) -> None:
        self.name = name
        self.severity = severity  # info | warning | critical
        self.message = message
        self.context = context
        self.fired_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return f"Alert({self.name}, {self.severity}): {self.message}"


class AlertManager:
    """
    Evaluates alert conditions and dispatches to registered handlers.
    Handlers receive (alert_name, severity, context).
    """

    def __init__(self) -> None:
        self._handlers: list[AlertHandler] = [self._log_handler]

    def register_handler(self, handler: AlertHandler) -> None:
        self._handlers.append(handler)

    def _dispatch(self, alert: Alert) -> None:
        for handler in self._handlers:
            try:
                handler(alert.name, alert.severity, alert.context)
            except Exception as exc:
                logger.error("alert_handler_error", handler=str(handler), error=str(exc))

    @staticmethod
    def _log_handler(name: str, severity: str, context: dict) -> None:
        log_fn = getattr(logger, severity if severity in ("info", "warning", "error") else "warning")
        log_fn("alert_fired", alert=name, **context)

    # ── Alert rules ─────────────────────────────────────────────────────────
    def check_low_confidence(self, signals: list[dict], threshold: float = 0.60) -> None:
        low = [s for s in signals if s.get("confidence", 1.0) < threshold]
        if len(low) / max(len(signals), 1) > 0.5:
            self._dispatch(Alert(
                name="low_signal_confidence",
                severity="warning",
                message=f"{len(low)}/{len(signals)} signals below confidence {threshold}",
                context={"low_confidence_signals": [s["ticker"] for s in low]},
            ))

    def check_model_staleness(self, last_trained_ts: datetime, max_hours: int = 48) -> None:
        age_hours = (datetime.now(timezone.utc) - last_trained_ts).total_seconds() / 3600
        if age_hours > max_hours:
            self._dispatch(Alert(
                name="model_stale",
                severity="warning",
                message=f"Signal model not retrained for {age_hours:.1f}h (threshold: {max_hours}h)",
                context={"last_trained": last_trained_ts.isoformat(), "age_hours": age_hours},
            ))

    def check_portfolio_drawdown(self, drawdown_pct: float, threshold_pct: float = 10.0) -> None:
        if abs(drawdown_pct) >= threshold_pct:
            severity = "critical" if abs(drawdown_pct) >= threshold_pct * 2 else "warning"
            self._dispatch(Alert(
                name="portfolio_drawdown",
                severity=severity,
                message=f"Portfolio drawdown {drawdown_pct:.1f}% exceeds threshold {threshold_pct:.1f}%",
                context={"drawdown_pct": drawdown_pct},
            ))

    def check_kafka_lag(self, topic: str, lag: int, threshold: int = 10_000) -> None:
        if lag > threshold:
            self._dispatch(Alert(
                name="kafka_consumer_lag",
                severity="warning",
                message=f"Kafka lag on {topic}: {lag} messages",
                context={"topic": topic, "lag": lag},
            ))
