# Inventory Optimization Business Logic

## Problem Statement

Retailers face two costly failure modes:

| Problem | Cost |
|---------|------|
| **Overstock** | Holding costs, spoilage, markdowns |
| **Stockout**  | Lost sales, customer churn, expedited shipping |

Industry data: Retailers lose **15–30% of potential revenue** to these two problems combined.

## Core Formulas

### 1. Safety Stock

```
Safety Stock = Z × σ_forecast_error × √(Lead Time in weeks)
```

| Variable | Meaning | Example |
|----------|---------|---------|
| Z | Z-score for service level | 1.65 (95%) · 2.33 (99%) |
| σ | Std dev of weekly forecast errors | Measured from actuals vs predictions |
| Lead Time | Weeks from order to delivery | 1 week (7 days) |

**Intuition:** Extra buffer to absorb demand spikes and supply delays with a controlled probability of stockout.

### 2. Reorder Point (ROP)

```
ROP = (Avg Weekly Demand × Lead Time in weeks) + Safety Stock
```

**Intuition:** When stock drops to this level, place an order. By the time it arrives, you'll have just enough to cover demand + the safety buffer.

**Example:**
- Avg demand = 100 units/week
- Lead time = 1 week
- Safety stock = 25 units
- ROP = (100 × 1) + 25 = **125 units**

When inventory hits 125 units → place order.

### 3. Economic Order Quantity (EOQ)

```
EOQ = √(2 × D × S / H)
```

| Variable | Meaning | Example |
|----------|---------|---------|
| D | Annual demand (units) | avg_weekly × 52 |
| S | Fixed cost per order ($) | $50 (shipping + admin) |
| H | Holding cost per unit per year ($) | unit_cost × 25% |

**Intuition:** Balances two competing costs:
- Ordering too often → high order costs (S)
- Ordering too much → high holding costs (H)
- EOQ is the quantity where these are equal → total cost is minimised

### 4. Total Annual Inventory Cost

```
Total Cost = (D / Q) × S  +  (Q / 2) × H
```

Where Q is the order quantity. At EOQ, this is minimised.

## Service Level Trade-offs

| Service Level | Z-score | Safety Stock | Customer Satisfaction |
|---------------|---------|-------------|----------------------|
| 90% | 1.28 | Lower | Risk of stockouts 10% of weeks |
| 95% | 1.65 | Medium | Risk 5% — **recommended default** |
| 99% | 2.33 | Higher | Risk 1% — high-cost items only |

## Expected Business Impact

A 95% service level with data-driven EOQ and ROP typically delivers:

- **20–30% reduction** in overstock holding costs
- **Stockout rate reduced** from ~15% to ~5% of weeks
- **Working capital freed** through right-sized safety stock
- **Fewer emergency orders** due to accurate ROP triggers
