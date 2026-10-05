"""Generate QuantLib reference prices for QuantJulia's cross-validation test.

    python -m pip install QuantLib==1.43
    python validation/quantlib_reference.py      # writes validation/quantlib_reference.csv

Every case uses flat continuously-compounded r and q, Actual/365 Fixed and a
maturity of an integer number of days, so T = days/365 exactly matches
QuantJulia's year fraction. The CSV is committed; test/test_quantlib.jl reads
it, so CI checks agreement without needing QuantLib installed.

Columns: case, model, kind, call, S, K, r, q, T, extra (semicolon-separated
model parameters), value.
"""
import csv
import QuantLib as ql

TODAY = ql.Date(5, 10, 2026)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()
CAL = ql.NullCalendar()
rows = []


def handles(S, r, q, vol=None):
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rf = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DC, ql.Continuous))
    dv = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, DC, ql.Continuous))
    bv = None
    if vol is not None:
        bv = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, CAL, vol, DC))
    return spot, rf, dv, bv


def option(K, days, call, american=False):
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, K)
    mat = TODAY + days
    ex = ql.AmericanExercise(TODAY, mat) if american else ql.EuropeanExercise(mat)
    return ql.VanillaOption(payoff, ex), mat


def add(case, model, kind, call, S, K, r, q, days, extra, value):
    rows.append([case, model, kind, int(call), S, K, r, q, days / 365,
                 ";".join(repr(float(x)) for x in extra), repr(float(value))])


S = 100.0
# --- Black-Scholes prices, Greeks, implied vol --------------------------------
for (K, days, r, q, vol) in [(80, 91, 0.03, 0.01, 0.25), (100, 182, 0.05, 0.0, 0.2),
                             (120, 365, 0.02, 0.02, 0.3), (95, 730, 0.04, 0.015, 0.18)]:
    for call in (True, False):
        spot, rf, dv, bv = handles(S, r, q, vol)
        proc = ql.BlackScholesMertonProcess(spot, dv, rf, bv)
        opt, _ = option(K, days, call)
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
        for kind, v in [("price", opt.NPV()), ("delta", opt.delta()), ("gamma", opt.gamma()),
                        ("vega", opt.vega()), ("theta", opt.theta()), ("rho", opt.rho())]:
            add("bs", "bs", kind, call, S, K, r, q, days, [vol], v)
        iv = opt.impliedVolatility(opt.NPV(), proc, 1e-12, 1000, 1e-4, 4.0)
        add("bs", "bs", "implied_vol", call, S, K, r, q, days, [vol], iv)

# --- Heston (AnalyticHestonEngine) ---------------------------------------------
for (kappa, theta, sigma, rho, v0) in [(2.0, 0.04, 0.5, -0.7, 0.04), (1.0, 0.04, 1.0, -0.9, 0.04),
                                       (3.0, 0.09, 0.3, -0.3, 0.06)]:
    for (K, days) in [(80, 91), (100, 182), (100, 730), (125, 365)]:
        for call in (True, False):
            spot, rf, dv, _ = handles(S, 0.03, 0.01)
            proc = ql.HestonProcess(rf, dv, spot, v0, kappa, theta, sigma, rho)
            opt, _ = option(K, days, call)
            opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc), 1e-12, 100000))
            add("heston", "heston", "price", call, S, K, 0.03, 0.01, days,
                [kappa, theta, sigma, rho, v0], opt.NPV())

# Merton jump-diffusion: QuantLib's Python bindings do not expose its
# JumpDiffusionEngine, so Merton is not cross-checked here (QuantJulia tests it
# against Merton's closed-form Poisson series instead); Bates below covers the
# same jump component on top of Heston.

# --- Bates (BatesEngine) -------------------------------------------------------
for (kappa, theta, sigma, rho, v0, lam, nu, delta) in [(2.0, 0.04, 0.5, -0.7, 0.04, 0.8, -0.12, 0.1)]:
    for (K, days) in [(85, 91), (100, 182), (110, 365)]:
        for call in (True, False):
            spot, rf, dv, _ = handles(S, 0.03, 0.01)
            proc = ql.BatesProcess(rf, dv, spot, v0, kappa, theta, sigma, rho, lam, nu, delta)
            opt, _ = option(K, days, call)
            opt.setPricingEngine(ql.BatesEngine(ql.BatesModel(proc), 1e-12, 100000))
            add("bates", "bates", "price", call, S, K, 0.03, 0.01, days,
                [kappa, theta, sigma, rho, v0, lam, nu, delta], opt.NPV())

# --- American put (finite differences, fine grid) -----------------------------------
for (K, days, r, q, vol) in [(100, 365, 0.05, 0.0, 0.2), (110, 182, 0.03, 0.01, 0.3), (90, 730, 0.06, 0.0, 0.25)]:
    spot, rf, dv, bv = handles(S, r, q, vol)
    proc = ql.BlackScholesMertonProcess(spot, dv, rf, bv)
    opt, _ = option(K, days, False, american=True)
    opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 2000, 2000))
    add("american", "bs", "price", False, S, K, r, q, days, [vol], opt.NPV())

# --- Barriers (AnalyticBarrierEngine, continuous monitoring, no rebate) ---------------
types = {"down_out": ql.Barrier.DownOut, "down_in": ql.Barrier.DownIn,
         "up_out": ql.Barrier.UpOut, "up_in": ql.Barrier.UpIn}
for kind, bt in types.items():
    H = 85.0 if kind.startswith("down") else 120.0
    for K in (90.0, 100.0, 110.0):
        for call in (True, False):
            spot, rf, dv, bv = handles(S, 0.05, 0.02, 0.25)
            proc = ql.BlackScholesMertonProcess(spot, dv, rf, bv)
            payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, K)
            opt = ql.BarrierOption(bt, H, 0.0, payoff, ql.EuropeanExercise(TODAY + 273))
            opt.setPricingEngine(ql.AnalyticBarrierEngine(proc))
            add("barrier", "bs", kind, call, S, K, 0.05, 0.02, 273, [0.25, H], opt.NPV())

# --- Geometric Asian, discrete fixings every 30 days -----------------------------------
for (K, n) in [(95.0, 12), (100.0, 12), (105.0, 6)]:
    for call in (True, False):
        days = 30 * n
        spot, rf, dv, bv = handles(S, 0.05, 0.01, 0.2)
        proc = ql.BlackScholesMertonProcess(spot, dv, rf, bv)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, K)
        fixings = [TODAY + 30 * i for i in range(1, n + 1)]
        opt = ql.DiscreteAveragingAsianOption(ql.Average.Geometric, 1.0, 0, fixings, payoff,
                                              ql.EuropeanExercise(TODAY + days))
        opt.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(proc))
        add("asian_geometric", "bs", "price", call, S, K, 0.05, 0.01, days, [0.2, n], opt.NPV())

with open("validation/quantlib_reference.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "model", "kind", "call", "S", "K", "r", "q", "T", "extra", "value"])
    w.writerows(rows)
print(f"wrote {len(rows)} reference values (QuantLib {ql.__version__})")
