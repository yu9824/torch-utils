from typing import Iterable, Tuple, Sequence, Callable
from typing import TypeVar, Union, overload, Literal, Optional, Any
import pkgutil
import inspect

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import matplotlib.axes
from sklearn.metrics import r2_score, mean_squared_error

T = TypeVar("T")


def is_argument(func: Callable[[Any], Any], arg_name: str) -> bool:
    """Check if a function has an argument

    Parameters
    ----------
    func : Callable[[Any], Any]
        function to check
    arg_name : str
        argument name to check

    Returns
    -------
    bool
        True if the function has the argument, False otherwise
    """
    return arg_name in inspect.signature(func).parameters.keys()


def is_installed(module_name: str) -> bool:
    """Check if module is installed

    Parameters
    ----------
    module_name : str
        module name to check

    Returns
    -------
    bool
        True if module is installed, False otherwise
    """
    return any(
        _module.name == module_name for _module in pkgutil.iter_modules()
    )


if is_installed("tqdm"):
    from tqdm.auto import tqdm

    @overload
    def check_tqdm(
        __iterable: Optional[Iterable[T]],
        silent: Literal[False],
        **kwargs,
    ) -> Union[tqdm, Iterable[T]]:
        ...

else:

    @overload
    def check_tqdm(
        __iterable: Optional[Iterable[T]],
        silent: Literal[False],
        **kwargs,
    ) -> Iterable[T]:
        ...


@overload
def check_tqdm(
    __iterable: Optional[Iterable[T]],
    silent: Literal[True],
    **kwargs,
) -> Iterable[T]:
    ...


def check_tqdm(
    __iterable: Optional[Iterable[T]] = None,
    silent: bool = False,
    **kwargs,
) -> Iterable[T]:
    if silent or not is_installed("tqdm"):
        return __iterable
    else:
        from tqdm.auto import tqdm

        return tqdm(__iterable, **kwargs)


class Limit:
    def __init__(self, *arrays: ArrayLike, alpha: float = 0.05) -> None:
        self.arrays = arrays
        self.alpha = alpha

        mins = []
        maxs = []
        for _arr in self.arrays:
            _arr = np.array(_arr).ravel()
            mins.append(np.min(_arr, axis=None))
            maxs.append(np.max(_arr, axis=None))

        self.__without_margin = (min(mins), max(maxs))
        offset = (self.__without_margin[1] - self.__without_margin[0]) * alpha
        self.__with_margin = (
            self.__without_margin[0] - offset,
            self.__without_margin[1] + offset,
        )

    @classmethod
    def from_with_margin(
        cls, with_margin: Tuple[float, float], alpha: float = 0.05
    ) -> "Limit":
        return cls(
            (
                (alpha * with_margin[1] + (1 + alpha) * with_margin[0])
                / (1 - alpha**2),
                (with_margin[1] + 2 * with_margin[0]) / (1 - alpha),
            ),
            alpha=alpha,
        )

    @property
    def without_margin(self) -> Tuple[float, float]:
        return self.__without_margin

    @without_margin.setter
    def without_margin(self, value: Tuple[float, float]) -> None:
        raise AttributeError("Cannot set attribute")

    @property
    def with_margin(self) -> Tuple[float, float]:
        return self.__with_margin

    @with_margin.setter
    def with_margin(self, value: Tuple[float, float]) -> None:
        raise AttributeError("Cannot set attribute")


def is_plotted(ax: Optional[matplotlib.axes.Axes] = None) -> bool:
    """Check if an axes has been plotted on.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        The axes to check. If None, the current axes will be used.
        optional, by default None

    Returns
    -------
    bool
        True if the axes has been plotted on, False otherwise.
    """
    ax = plt.gca() if ax is None else ax
    return any(
        len(getattr(ax, _key))
        for _key in dir(ax)
        if isinstance(getattr(ax, _key), matplotlib.axes.Axes.ArtistList)
    )


def root_mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: str = "uniform_average",
) -> float:
    return mean_squared_error(
        y_true,
        y_pred,
        squared=False,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


def get_score_txt(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    score_mapping: dict[str, Callable[[ArrayLike, ArrayLike], float]] = {
        "$\\mathrm{RMSE}$": root_mean_squared_error,
        "$R^2$": r2_score,
    },
    subscript: Optional[str] = None,
    fmt: str = ".2g",
) -> str:
    subscript = "" if subscript is None else subscript
    score_txts = []
    for name, score_func in score_mapping.items():
        score = score_func(y_true, y_pred)
        score_txts.append(f"{name}$_\\mathrm{{{subscript}}}$: {score:{fmt}}")
    return "\n".join(score_txts)


def yyplot(
    *y: ArrayLike,
    ax: Optional[matplotlib.axes.Axes] = None,
    labels: Optional[Sequence[str]] = None,
    show_score: bool = True,
    **kwargs_scatter,
) -> matplotlib.axes.Axes:
    if is_plotted(ax=ax) and ax is None:
        fig, ax = plt.subplots(facecolor="w", figsize=(3.2, 3.2), dpi=300)
    else:
        ax = plt.gca() if ax is None else ax
        fig = ax.figure

    assert len(y) % 2 == 0, "Number of yays must be even."

    n_pairs = len(y) // 2
    if labels is None:
        if n_pairs == 2:
            labels = ["train", "test"]
        elif n_pairs == 3:
            labels = ["train", "val", "test"]
        else:
            labels = None
    else:
        assert len(y) // 2 == len(
            labels
        ), "Number of labels must be half of arrays."

    lim = Limit(*y)
    if show_score:
        txts = []
    for i in range(n_pairs):
        y_true, y_pred = y[2 * i], y[2 * i + 1]
        ax.scatter(
            y_true,
            y_pred,
            label=labels[i] if labels else None,
            **kwargs_scatter,
        )

        if show_score and labels is not None:
            _txt = get_score_txt(
                y_true, y_pred, subscript=labels[i] if labels else None
            )
            txts.append(_txt)

    if show_score:
        ax.text(
            lim.without_margin[0],
            lim.without_margin[1],
            "\n".join(txts),
            ha="left",
            va="top",
            bbox=dict(
                facecolor="#f0f0f0",
                alpha=0.7,
                edgecolor="None",
                boxstyle="round",
            ),
        )

    # layout
    ax.plot(*[lim.with_margin] * 2, color="gray", zorder=0.5)
    ax.set_xlim(lim.with_margin)
    ax.set_ylim(lim.with_margin)
    ax.set_xlabel("$y_\\mathrm{true}$")
    ax.set_ylabel("$y_\\mathrm{pred}$")
    ax.set_aspect("equal", adjustable="box")

    if labels is not None:
        ax.legend(loc="lower right")
    fig.tight_layout()
    return ax


if __name__ == "__main__":
    # limit = Limit(np.arange(12), np.arange(12))
    # print(limit.arrays)
    yyplot(*[np.random.rand(12) for _ in range(4)])
    plt.show()
