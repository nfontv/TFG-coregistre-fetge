"""

Optimitzadors previstos (els 3 dels experiments + variants):
    - 'powell'      -> scipy.optimize.minimize(method='Powell')   [derivative-free]
    - 'nelder-mead' -> scipy.optimize.minimize(method='Nelder-Mead')
    - 'lbfgsb'      -> scipy.optimize.minimize(method='L-BFGS-B')  [gradient numeric]
    - 'cmaes'       -> llibreria cma (pip install cma)             [evolutiu]

"""

from dataclasses import dataclass, field
from typing import Callable
import time

import numpy as np
from scipy.optimize import minimize
import cma


# Resultat unificat: passi el que passi per sota, el run_vell.py rep aixo.

@dataclass
class ResultatOptim:
    x: np.ndarray  # parametres òptims (6,)
    fun: float     # valor final de la pèrdua
    n_iter: int
    n_feval: int
    converged: bool # l'optimitzador ha ha convergit o no
    time_s: float   # temps computacional
    history: list = field(default_factory=list, repr=False)  # valor de la perdua a cada avaluacio
    raw: object = field(default=None, repr=False)  # objecte per si es vol vols inspeccionar


# Conjunt de mètodes que passen directament per scipy.optimize.minimize.
_SCIPY_METHODS = {"powell": "Powell",
                  "lbfgsb": "L-BFGS-B"}


def run_optimizer(
    loss_fn: Callable[[np.ndarray], float], #funcio de perdua
    x0: np.ndarray, #inicialització per centroides
    optimizer: str, #'powell', 'nelder-mead', 'lbfgsb' , 'cmaes'
    maxiter: int = 100,
    bounds=None, #llista de (min,max) per paràmetre
    sigma0: float = None, #necessari per cmaes, desviacio inicial per la cerca
    seed: int = 0, #necessari per cmaes
    monitor_fn = None,
):

    t0 = time.perf_counter()
    history = []

    def loss_fn_registrada(params):
        valor = loss_fn(params)
        if monitor_fn is None:
            history.append(valor)
        else:
            history.append((valor, float(monitor_fn(params))))
        return valor

    if optimizer in _SCIPY_METHODS:
        result = _run_scipy(loss_fn_registrada, x0, _SCIPY_METHODS[optimizer], maxiter, bounds)
        elapsed = time.perf_counter() - t0
        return ResultatOptim(
            x=np.asarray(result.x, dtype=float),
            fun=float(result.fun),
            n_iter=int(getattr(result, "nit", -1)),
            n_feval=int(getattr(result, "nfev", -1)),
            converged=bool(result.success),
            time_s=elapsed,
            history=history,
            raw=result,
        )

    if optimizer == "cmaes":
        result = _run_cmaes(loss_fn_registrada, x0, maxiter, bounds, sigma0, seed)
        result.time_s = time.perf_counter() - t0
        result.history = history
        return result

    raise ValueError(f"Optimitzador desconegut: {optimizer!r}. "
                     f"Opcions: {list(_SCIPY_METHODS) + ['cmaes']}")


def _run_scipy(loss_fn, x0, method, maxiter, bounds):
    """
    Powell es derivative-free; L-BFGS-B aproxima el gradient per diferencies finites .
    Queda pendent determinar maxiter, bounds, etc
    """
    options = {"maxiter": maxiter, "disp": False}

    if method == "L-BFGS-B":
        options["eps"] = np.array([0.3, 0.3, 0.3, 0.01, 0.01, 0.01])  # pas de diferencies finites; prova 0.1-0.5
    # L-BFGS-B accepta bounds
    kwargs = {} #arguments amb nom, no importa el ordre
    if bounds is not None and method in ("L-BFGS-B", "Powell"):
        kwargs["bounds"] = bounds

    return minimize(fun=loss_fn, x0=np.asarray(x0, dtype=float),
                    method=method, options=options, **kwargs)


def _run_cmaes(loss_fn, x0, maxiter, bounds, sigma0, seed):

    if sigma0 is None:
        #pendent de triarn-e un adequat
        sigma0 = 1.0

    # escalat per dimensio. Si poses bounds, cma els pot usar.
    opts = {
        "maxiter": maxiter,
        "seed": seed,
        "verbose": -9,   # silenci
        "CMA_stds": [7.0, 7.0, 7.0, 0.05, 0.05, 0.05],
    }

    if bounds is not None:
        lows = [b[0] for b in bounds]
        highs = [b[1] for b in bounds]
        opts["bounds"] = [lows, highs]

    es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=float).tolist(), sigma0, opts)
    es.optimize(loss_fn)

    res = es.result  # namedtuple: xbest, fbest, evals_best, evaluations, iterations, ...
    return ResultatOptim(
        x=np.asarray(res.xbest, dtype=float),
        fun=float(res.fbest),
        n_iter=int(res.iterations),
        n_feval=int(res.evaluations),
        converged=bool(es.stop()),   # dict no buit -> ha parat per algun criteri
        time_s=0.0,                  # l'omple run_optimizer
        raw=res,
    )