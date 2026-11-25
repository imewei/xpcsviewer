"""Bayesian diagnosis window for MCMC fit results.

Provides a shared diagnostic window used by both the G2 Fitting tab
and the Diffusion tab.  Contains:

- Tab 1: Posterior predictive plot with 95% credible interval
- Tab 2: Six ArviZ diagnostic plots (pair, forest, energy, autocorr, rank, ESS)
- Bottom panel: Convergence summary (R-hat, ESS, divergences, BFMI)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
)

from xpcsviewer.gui.qt_compat import (
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    Qt,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from xpcsviewer.utils.export_guard import ExportCollisionError, ExportGuard

if TYPE_CHECKING:
    from xpcsviewer.fitting.results import FitDiagnostics, FitResult

logger = logging.getLogger(__name__)

# NOTE: Do not call matplotlib.use() here — backend is managed by
# plothandler/matplot_qt.py or the application entry point.


# ---------------------------------------------------------------------------
# What a window shows
# ---------------------------------------------------------------------------

_ANY: Any = object()  # "any bin" for BayesianDiagnosisWindow.shows


@dataclass(frozen=True, eq=False)
class DiagnosisIdentity:
    """The identity of what a diagnosis window shows.

    ``path`` / ``revision``: the data file and its on-disk revision the
    posterior was fitted from; ``bin_id``: absolute bin ID (``None`` for a
    diffusion fit); ``bundle``: the posterior with the observations it was
    fitted to.  Invalidation and navigation ask the window about this identity,
    never a store that may hold a newer object by now.
    """

    path: str | None
    revision: tuple[int, ...] | None
    bin_id: Hashable | None
    bundle: Any = None


# ---------------------------------------------------------------------------
# Convergence summary widget
# ---------------------------------------------------------------------------


class ConvergenceSummaryWidget(QWidget):
    """Rich-text summary of MCMC convergence diagnostics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        self._label = QLabel(self)
        self._label.setWordWrap(True)
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setText("<i>No results yet</i>")
        layout.addWidget(self._label)

    def clear(self) -> None:
        self._label.setText("<i>No results yet</i>")

    def update_diagnostics(self, diag: FitDiagnostics) -> None:
        """Refresh the summary from a ``FitDiagnostics`` instance."""
        rows: list[str] = []

        # Overall status
        status = (
            "<span style='color:green;font-weight:bold;'>CONVERGED</span>"
            if diag.converged
            else "<span style='color:red;font-weight:bold;'>NOT CONVERGED</span>"
        )
        rows.append(f"<b>Status:</b> {status}")

        # R-hat
        if diag.r_hat:
            rhat_parts = [f"{k}={v:.4f}" for k, v in diag.r_hat.items()]
            rows.append(f"<b>R-hat:</b> {', '.join(rhat_parts)}")

        # ESS bulk
        if diag.ess_bulk:
            ess_parts = [f"{k}={v}" for k, v in diag.ess_bulk.items()]
            rows.append(f"<b>ESS bulk:</b> {', '.join(ess_parts)}")

        # ESS tail
        if diag.ess_tail:
            ess_parts = [f"{k}={v}" for k, v in diag.ess_tail.items()]
            rows.append(f"<b>ESS tail:</b> {', '.join(ess_parts)}")

        # Divergences
        rows.append(f"<b>Divergences:</b> {diag.divergences}")

        # BFMI
        if diag.bfmi is not None:
            bfmi_color = "green" if diag.bfmi >= 0.2 else "red"
            rows.append(
                f"<b>BFMI:</b> <span style='color:{bfmi_color};'>{diag.bfmi:.3f}</span>"
            )
        else:
            # A missing BFMI fails convergence; say so rather than hide the row.
            rows.append(
                "<b>BFMI:</b> <span style='color:red;'>unavailable "
                "(fit not converged)</span>"
            )

        self._label.setText("<br>".join(rows))


def _norm(path: str | None) -> str | None:
    return None if path is None else os.path.normpath(path)


# ---------------------------------------------------------------------------
# Main diagnosis window
# ---------------------------------------------------------------------------


class BayesianDiagnosisWindow(QMainWindow):
    """Diagnostic window showing Bayesian fit results and MCMC diagnostics.

    Reused for both G2 single-Q fits and diffusion power-law fits by
    calling :meth:`update_results` with different data/labels.
    """

    _ARVIZ_PLOT_NAMES = ("pair", "forest", "energy", "autocorr", "rank", "ess")

    def __init__(
        self, parent: QWidget | None = None, title: str = "Bayesian Diagnosis"
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 750)

        # Central widget
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Splitter: top = tabs, bottom = convergence summary
        splitter = QSplitter(Qt.Orientation.Vertical, central)
        main_layout.addWidget(splitter)

        # --- Tab widget ---
        self._tabs = QTabWidget()
        splitter.addWidget(self._tabs)

        # Tab 1: Posterior predictive
        self._pp_widget = QWidget()
        pp_layout = QVBoxLayout(self._pp_widget)
        self._pp_canvas = FigureCanvasQTAgg(Figure(figsize=(7, 4)))
        pp_layout.addWidget(self._pp_canvas)
        # Export buttons for Tab 1
        pp_btn_row = QHBoxLayout()
        pp_btn_row.addStretch()
        self._btn_export_fit_plot = QPushButton("Export Plot")
        self._btn_export_fit_plot.setToolTip("Save fit + 95% CI plot as PNG/PDF")
        self._btn_export_fit_plot.clicked.connect(self._export_fit_plot)
        pp_btn_row.addWidget(self._btn_export_fit_plot)
        self._btn_export_fit_data = QPushButton("Export Data")
        self._btn_export_fit_data.setToolTip(
            "Save raw data, fitted curve, 95% CI, and residuals as CSV"
        )
        self._btn_export_fit_data.clicked.connect(self._export_fit_data)
        pp_btn_row.addWidget(self._btn_export_fit_data)
        pp_layout.addLayout(pp_btn_row)
        self._tabs.addTab(self._pp_widget, "Fit + 95% CI")

        # Tab 2: ArviZ diagnostics (2x3 grid inside scroll area)
        diag_container = QWidget()
        diag_container_layout = QVBoxLayout(diag_container)
        diag_container_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._diag_widget = QWidget()
        self._diag_layout = QGridLayout(self._diag_widget)
        self._diag_layout.setContentsMargins(4, 4, 4, 4)
        self._diag_layout.setSpacing(6)
        self._arviz_canvases: dict[str, FigureCanvasQTAgg] = {}
        for idx, name in enumerate(self._ARVIZ_PLOT_NAMES):
            canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4)))
            canvas.setMinimumSize(420, 360)
            canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            row, col = divmod(idx, 3)
            self._diag_layout.addWidget(canvas, row, col)
            self._arviz_canvases[name] = canvas
        scroll.setWidget(self._diag_widget)
        diag_container_layout.addWidget(scroll)
        # Export buttons for Tab 2
        diag_btn_row = QHBoxLayout()
        diag_btn_row.addStretch()
        self._btn_export_diag_plots = QPushButton("Export Plots")
        self._btn_export_diag_plots.setToolTip(
            "Save all 6 diagnostic plots as individual PNGs"
        )
        self._btn_export_diag_plots.clicked.connect(self._export_diag_plots)
        diag_btn_row.addWidget(self._btn_export_diag_plots)
        self._btn_export_traces = QPushButton("Export Traces")
        self._btn_export_traces.setToolTip(
            "Save ArviZ data as netCDF for further analysis"
        )
        self._btn_export_traces.clicked.connect(self._export_traces)
        diag_btn_row.addWidget(self._btn_export_traces)
        diag_container_layout.addLayout(diag_btn_row)
        self._tabs.addTab(diag_container, "MCMC Diagnostics")

        # --- Convergence summary ---
        self._summary = ConvergenceSummaryWidget()
        splitter.addWidget(self._summary)

        # Give most space to tabs
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)

        # Track the init-time figures so closeEvent can clean them up
        # even if _update_arviz_diagnostics replaces the canvas figures.
        self._original_figures = [self._pp_canvas.figure] + [
            c.figure for c in self._arviz_canvases.values()
        ]

        # Store axis labels (customisable per context)
        self._xlabel = "x"
        self._ylabel = "y"
        # Store latest results for export
        self._result: FitResult | None = None
        self._model_func: Any = None
        self._x_data: np.ndarray | None = None
        self._y_data: np.ndarray | None = None
        self._yerr: np.ndarray | None = None
        self._q_value: float | None = None
        self._identity: DiagnosisIdentity | None = None
        self._set_exports_enabled(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def identity(self) -> DiagnosisIdentity | None:
        """What the window shows now; ``None`` when it is empty."""
        return self._identity

    def shows(
        self,
        path: str | None,
        revision: tuple[int, ...] | None,
        bin_id: Hashable | None = _ANY,
    ) -> bool:
        """True if the window shows a posterior of ``path`` at ``revision``
        (and, when given, of absolute bin ``bin_id``)."""
        ident = self._identity
        if ident is None or self._result is None:
            return False
        if _norm(ident.path) != _norm(path) or ident.revision != revision:
            return False
        return bin_id is _ANY or ident.bin_id == bin_id

    def matches(self, predicate: Callable[[str | None, tuple | None], bool]) -> bool:
        """True if the window shows something whose ``(path, revision)``
        satisfies ``predicate``."""
        ident = self._identity
        return (
            ident is not None
            and self._result is not None
            and bool(predicate(ident.path, ident.revision))
        )

    def clear(self, title: str = "Bayesian Diagnosis") -> None:
        """Explicit empty state: drop the result, empty every canvas, disable
        the exports and retitle (nothing of the old posterior stays visible)."""
        self._identity = None
        self._result = None
        self._model_func = None
        self._x_data = self._y_data = self._yerr = None
        self._q_value = None
        self._pp_canvas.figure.clear()
        self._pp_canvas.draw()
        for canvas in self._arviz_canvases.values():
            canvas.figure.clear()
            canvas.draw()
        self._summary.clear()
        self._set_exports_enabled(False)
        self.setWindowTitle(title)

    def _set_exports_enabled(self, enabled: bool) -> None:
        for button in (
            self._btn_export_fit_plot,
            self._btn_export_fit_data,
            self._btn_export_diag_plots,
            self._btn_export_traces,
        ):
            button.setEnabled(enabled)

    def set_axis_labels(self, xlabel: str, ylabel: str) -> None:
        """Set axis labels for the posterior predictive plot."""
        self._xlabel = xlabel
        self._ylabel = ylabel

    def update_results(
        self,
        result: FitResult,
        model_func: Any,
        x_data: Any,
        y_data: Any,
        yerr: Any | None = None,
        q_value: float | None = None,
        title: str | None = None,
        *,
        identity: DiagnosisIdentity,
    ) -> None:
        """Refresh all panels with new fit results.

        Parameters
        ----------
        result : FitResult
            Bayesian fit result with posterior samples and diagnostics.
        model_func : callable
            Model function ``f(x, *params)`` for posterior predictive.
        x_data, y_data : array-like
            Original data used for fitting.
        yerr : array-like, optional
            Measurement uncertainties (shown as errorbars if provided).
        q_value : float, optional
            Q value for display in the title.
        title : str, optional
            Window title override.
        identity : DiagnosisIdentity
            What the result belongs to (file, revision, bin, bundle); the
            window answers :meth:`shows` / :meth:`matches` from it.
        """
        if title is not None:
            self.setWindowTitle(title)

        self._identity = identity
        self._set_exports_enabled(True)
        self._result = result
        self._model_func = model_func
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        self._yerr = np.asarray(yerr) if yerr is not None else None
        self._q_value = q_value
        self._update_posterior_predictive(
            result, model_func, x_data, y_data, yerr, q_value
        )
        self._update_arviz_diagnostics(result)
        self._summary.update_diagnostics(result.diagnostics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_posterior_predictive(
        self,
        result: FitResult,
        model_func: Any,
        x_data: Any,
        y_data: Any,
        yerr: Any | None,
        q_value: float | None,
    ) -> None:
        """Redraw the posterior predictive plot (Tab 1)."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        fig = self._pp_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Plot data with errorbars if available
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        if yerr is not None:
            yerr = np.asarray(yerr)
            ax.errorbar(
                x_data,
                y_data,
                yerr=yerr,
                fmt="ko",
                ms=4,
                alpha=0.7,
                label="Data",
                zorder=3,
            )
        else:
            ax.scatter(x_data, y_data, c="k", s=20, alpha=0.7, label="Data", zorder=3)

        # Log-spaced prediction for smooth curve on log-x axis
        pos = x_data[x_data > 0]
        if len(pos) < 2:
            x_pred = np.linspace(x_data.min(), x_data.max(), 200)
        else:
            x_pred = np.geomspace(pos.min(), pos.max(), 200)

        plot_posterior_predictive(
            result,
            model_func,
            x_data,
            y_data,
            x_pred=x_pred,
            ax=ax,
        )

        # Remove the duplicate "Data" scatter that plot_posterior_predictive adds
        handles, labels = ax.get_legend_handles_labels()
        seen: set[str] = set()
        unique_handles, unique_labels = [], []
        for h, lbl in zip(handles, labels, strict=False):
            if lbl not in seen:
                seen.add(lbl)
                unique_handles.append(h)
                unique_labels.append(lbl)
        ax.legend(unique_handles, unique_labels)

        ax.set_xscale("log")
        ax.set_xlabel(self._xlabel)
        ax.set_ylabel(self._ylabel)
        subtitle = f"Q = {q_value:.4g}" if q_value is not None else ""
        ax.set_title(f"Posterior Predictive {subtitle}".strip())

        fig.tight_layout()
        self._pp_canvas.draw()

    def _update_arviz_diagnostics(self, result: FitResult) -> None:
        """Redraw ArviZ diagnostic plots (Tab 2)."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        if result.arviz_data is None:
            logger.warning("No ArviZ data available — skipping diagnostic plots")
            return

        import matplotlib.pyplot as plt

        var_names = list(result.samples.keys())
        try:
            figures = generate_arviz_diagnostics(result.arviz_data, var_names=var_names)
        except Exception:
            logger.exception("Failed to generate ArviZ diagnostics")
            return

        for name, canvas in self._arviz_canvases.items():
            fig = figures.get(name)
            if fig is None:
                canvas.figure.clear()
                ax = canvas.figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    f"{name}\n(not available)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                canvas.draw()
                continue

            # Close the old figure before replacing to prevent leak
            old_fig = canvas.figure
            canvas.figure = fig
            fig.set_canvas(canvas)
            # Reduce font sizes so labels fit within the canvas
            for ax in fig.get_axes():
                ax.tick_params(labelsize=8)
                ax.xaxis.label.set_size(9)
                ax.yaxis.label.set_size(9)
                if ax.get_title():
                    ax.title.set_size(9)
            fig.tight_layout(pad=1.2, h_pad=1.0, w_pad=1.0)
            canvas.draw()
            plt.close(old_fig)

    # ------------------------------------------------------------------
    # Export handlers
    # ------------------------------------------------------------------

    def _new_guard(self) -> ExportGuard:
        """One guard per export action; the source data file is an input."""
        ident = self._identity
        return ExportGuard([ident.path] if ident is not None and ident.path else [])

    def _claim(self, guard: ExportGuard, path: Any, what: str) -> bool:
        """Claim ``path``; a refusal is logged and shown, nothing is written."""
        try:
            guard.claim(path, what)
        except ExportCollisionError as exc:
            logger.error("Export refused: %s", exc)
            self.statusBar().showMessage(f"Export refused: {exc}", 10000)
            return False
        return True

    def _export_fit_plot(self) -> None:
        """Save the Fit + 95% CI plot to file."""
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Plot",
            "bayesian_fit.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        if not self._claim(self._new_guard(), path, "fit plot"):
            return
        fig = self._pp_canvas.figure
        note = None
        if self._result is not None and not self._result.diagnostics.converged:
            # a posterior that did not converge is not a measurement
            note = fig.text(
                0.5, 0.995, "NOT CONVERGED - not a measurement", color="red", va="top"
            )
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        finally:
            if note is not None:
                note.remove()
        logger.info("Exported fit plot to %s", path)

    def _export_fit_data(self) -> None:
        """Save raw data, fitted curve, 95% CI, and residuals as CSV."""
        if (
            self._result is None
            or self._x_data is None
            or self._y_data is None
            or self._model_func is None
        ):
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Data",
            "bayesian_fit.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        if not self._claim(self._new_guard(), path, "fit data"):
            return
        import csv
        from pathlib import Path

        result = self._result
        param_names = result.param_names or list(result.samples.keys())
        if not param_names:
            logger.error("Cannot export fit data: no sampled parameters available")
            return
        n_samples = len(result.samples[param_names[0]])
        x = self._x_data
        model = self._model_func

        # Compute posterior predictions at each data point
        predictions = np.empty((n_samples, len(x)))
        for i in range(n_samples):
            params = [result.samples[name][i] for name in param_names]
            predictions[i] = model(x, *params)

        y_fit = np.median(predictions, axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        residual = self._y_data - y_fit

        with Path(path).open("w", newline="") as f:
            writer = csv.writer(f)
            # Header with metadata comment
            f.write(f"# Q = {self._q_value}\n" if self._q_value is not None else "")
            if not result.diagnostics.converged:
                f.write("# NOT CONVERGED: not a measurement (diagnosis only)\n")
            header = [
                self._xlabel,
                self._ylabel,
                "yerr",
                "fit_median",
                "ci_lower_2.5%",
                "ci_upper_97.5%",
                "residual",
            ]
            writer.writerow(header)
            for j in range(len(x)):
                row = [
                    f"{x[j]:.6g}",
                    f"{self._y_data[j]:.6g}",
                    f"{self._yerr[j]:.6g}" if self._yerr is not None else "",
                    f"{y_fit[j]:.6g}",
                    f"{ci_lower[j]:.6g}",
                    f"{ci_upper[j]:.6g}",
                    f"{residual[j]:.6g}",
                ]
                writer.writerow(row)

        logger.info("Exported fit data (%d points) to %s", len(x), path)

    def _export_diag_plots(self) -> None:
        """Save all 6 diagnostic plots as individual files."""
        from pathlib import Path

        if self._result is None:
            return
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not dir_path:
            return
        out = Path(dir_path)
        guard = self._new_guard()
        written = 0
        for name, canvas in self._arviz_canvases.items():
            fig = canvas.figure
            filepath = out / f"mcmc_{name}.png"
            if not self._claim(guard, filepath, f"{name} plot"):
                continue
            fig.savefig(str(filepath), dpi=300, bbox_inches="tight")
            written += 1
        logger.info("Exported %d diagnostic plots to %s", written, out)

    def _export_traces(self) -> None:
        """Save ArviZ DataTree as netCDF."""
        if self._result is None or self._result.arviz_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Traces (netCDF)",
            "mcmc_traces.nc",
            "netCDF (*.nc)",
        )
        if not path:
            return
        if not self._claim(self._new_guard(), path, "traces"):
            return
        # One rule with the All-Q writer (fitting.viz.export_bayesian_diagnostics):
        # a file carries ``bin_status`` (and the bin's identity). The All-Q export
        # leaves non-converged posteriors out; this window exists to DIAGNOSE one,
        # so it writes it, but only marked ``not_converged``, and says so.
        from xpcsviewer.fitting.viz import _tagged

        converged = bool(self._result.diagnostics.converged)
        attrs: dict[str, Any] = {"bin_status": "ok" if converged else "not_converged"}
        ident = self._identity
        if ident is not None and ident.bin_id is not None:
            try:
                attrs["bin_id"] = int(ident.bin_id)  # type: ignore[call-overload]
            except (TypeError, ValueError):
                pass
        if self._q_value is not None:
            attrs["q_value"] = float(self._q_value)
        data = _tagged(self._result.arviz_data, attrs)
        if getattr(data, "attrs", {}).get("bin_status") != attrs["bin_status"]:
            # the posterior could not be tagged: never write an unmarked file
            logger.error("Export Traces refused: the posterior cannot be tagged")
            self.statusBar().showMessage(
                "Export refused: the traces cannot be marked with their status", 10000
            )
            return
        try:
            data.to_netcdf(path)
        except ValueError as e:  # xarray: no netCDF backend installed
            logger.warning("netCDF export needs netCDF4 or h5netcdf: %s", e)
            self.setWindowTitle(
                f"{self.windowTitle()} — netCDF export needs netCDF4 or h5netcdf"
            )
            return
        logger.info("Exported ArviZ traces to %s (%s)", path, attrs["bin_status"])
        if not converged:
            self.statusBar().showMessage(
                "Traces exported and marked bin_status=not_converged: "
                "not a measurement (diagnosis only)",
                10000,
            )

    def closeEvent(self, event: Any) -> None:
        """Clean up matplotlib figures to avoid memory leaks."""
        import matplotlib.pyplot as plt

        # Close current canvas figures
        for canvas in self._arviz_canvases.values():
            plt.close(canvas.figure)
        plt.close(self._pp_canvas.figure)

        # Also close init-time figures that may have been swapped out
        for fig in self._original_figures:
            plt.close(fig)

        super().closeEvent(event)
