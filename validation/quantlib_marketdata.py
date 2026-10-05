"""QuantLib 1.43 references for QuantJulia's market-data layer.

    python validation/quantlib_marketdata.py   # writes validation/quantlib_marketdata.csv

Zero curves are ZeroCurve(dates, rates, Actual365Fixed, NullCalendar, Linear,
Continuous) — zero rates linear in time, as QuantJulia's ZeroCurve. Cash
dividends use the escrowed model: AnalyticDividendEuropeanEngine for
Europeans, FdBlackScholesVanillaEngine with CashDividendModel Escrowed for
Americans. Columns: case, call, K, T, value.
"""
import csv
import QuantLib as ql

TODAY = ql.Date(5, 10, 2026)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()
S = 100.0
PILLAR_DAYS = [0, 30, 182, 365, 730, 1825]
R_ZEROS = [0.030, 0.032, 0.036, 0.040, 0.042, 0.045]
Q_ZEROS = [0.010, 0.010, 0.012, 0.015, 0.015, 0.016]
DIV_DAYS, DIV_AMTS = [60, 150, 240, 330], [0.8, 0.8, 0.9, 0.9]


def curve(zeros):
    dates = [TODAY + d for d in PILLAR_DAYS]
    c = ql.ZeroCurve(dates, zeros, DC, ql.NullCalendar(), ql.Linear(), ql.Continuous)
    return ql.YieldTermStructureHandle(c)


rf, dv = curve(R_ZEROS), curve(Q_ZEROS)
spot = ql.QuoteHandle(ql.SimpleQuote(S))
rows = []

# discount factors at off-pillar dates
for d in (15, 100, 300, 500, 1000, 1500):
    rows.append(["discount_r", 1, 0.0, d / 365, rf.discount(TODAY + d)])
    rows.append(["discount_q", 1, 0.0, d / 365, dv.discount(TODAY + d)])

vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), 0.22, DC))
proc = ql.BlackScholesMertonProcess(spot, dv, rf, vol)


def vanilla(K, days, call, american=False):
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, K)
    ex = ql.AmericanExercise(TODAY, TODAY + days) if american else ql.EuropeanExercise(TODAY + days)
    return ql.VanillaOption(payoff, ex)


# Black-Scholes on term-structure curves (no cash dividends)
for K, days in [(90.0, 120), (100.0, 400), (115.0, 900)]:
    for call in (True, False):
        o = vanilla(K, days, call)
        o.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
        rows.append(["bs_curves", int(call), K, days / 365, o.NPV()])

# escrowed cash dividends, European and American
divs = ql.DividendVector([TODAY + d for d in DIV_DAYS], DIV_AMTS)
for K, days in [(95.0, 200), (100.0, 365), (105.0, 365)]:
    for call in (True, False):
        o = vanilla(K, days, call)
        o.setPricingEngine(ql.AnalyticDividendEuropeanEngine(proc, divs))
        rows.append(["bs_cashdiv_european", int(call), K, days / 365, o.NPV()])
    o = vanilla(K, days, False, american=True)
    o.setPricingEngine(ql.FdBlackScholesVanillaEngine(
        proc, divs, 1000, 1000, 0, ql.FdmSchemeDesc.Douglas(), False, -ql.nullDouble(),
        ql.FdBlackScholesVanillaEngine.Escrowed))
    rows.append(["bs_cashdiv_american_put", 0, K, days / 365, o.NPV()])

# Heston on term-structure curves
hp = ql.HestonProcess(rf, dv, spot, 0.04, 2.0, 0.04, 0.5, -0.7)    # v0, κ, θ, ξ, ρ
for K, days in [(85.0, 90), (100.0, 365), (120.0, 730)]:
    for call in (True, False):
        o = vanilla(K, days, call)
        o.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(hp), 1e-12, 100000))
        rows.append(["heston_curves", int(call), K, days / 365, o.NPV()])

with open("validation/quantlib_marketdata.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "call", "K", "T", "value"])
    w.writerows([[c, k, K, repr(T), repr(float(v))] for c, k, K, T, v in rows])
print(f"wrote {len(rows)} market-data reference values (QuantLib {ql.__version__})")
