from pricing.margin import compute_margin

ABSOLUTE_MIN_PRICE = 20_000
SUPPLIER_MULTIPLIER = 1.72
COLD_STORAGE_FACTOR = 100_000

def round_to_nearest(value, base=5000):
    return int(round(value / base) * base)

def calculate_sale_price(
    supplier_price,
    market_price=None,
    cold_storage=25,
    pricing_mode="market",
    manual_margin=0.30
):
    coste_interno = supplier_price * SUPPLIER_MULTIPLIER

    if pricing_mode == "market" and market_price:
        market_ratio = market_price / coste_interno
        margin = compute_margin(market_ratio)
    else:
        margin = max(manual_margin, 0.20)
        market_ratio = None

    price = coste_interno * (1 + margin)

    if cold_storage <= 10:
        price += COLD_STORAGE_FACTOR

    price = max(price, ABSOLUTE_MIN_PRICE)

    return {
        "Coste proveedor": round(supplier_price, 2),
        "Coste interno": round(coste_interno, 2),
        "Precio de mercado": market_price,
        "Market ratio": None if market_ratio is None else f'{int(100*market_ratio)}%',
        "Margen aplicado": f'{int(100*margin)}%',
        "Precio de venta": round_to_nearest(price)
    }
