# Example Usage

```python
import polars as pl
from sketches import ThetaSketch

# Read data (adjust the path as needed)
df = pl.read_csv("../../tests/data/customer.csv")

# Select column
column = "c_custkey"

# Partition data into two subsets (e.g., even and odd keys)
df1 = df.filter(pl.col(column) % 2 == 0)
df2 = df.filter(pl.col(column) % 2 != 0)

values1 = df1[column].cast(str).to_list()
values2 = df2[column].cast(str).to_list()

# Create and populate the sketches
s1 = ThetaSketch()
s2 = ThetaSketch()
for v in values1:
    s1.update(v)
for v in values2:
    s2.update(v)

# Actual counts using Polars
actual_union = pl.concat([df1.select(column), df2.select(column)]).unique().height
actual_intersection = (
    df1.select(column).join(df2.select(column), on=column, how="inner").height
)
actual_difference = (
    df1.select(column).join(df2.select(column), on=column, how="anti").height
)

# Sketch estimates
union = s1.union(s2)
intersection = s1.intersect(s2)
difference = s1.difference(s2)

# Print results
print(f"Actual union size: {actual_union}")
print(f"Estimated union size (Theta): {union.estimate():.2f}")
print(f"Actual intersection size: {actual_intersection}")
print(f"Estimated intersection size (Theta): {intersection.estimate():.2f}")
print(f"Actual difference size: {actual_difference}")
print(f"Estimated difference size (Theta): {difference.estimate():.2f}")
# Actual union size: 150000
# Estimated union size (Theta): 149420.83
# Actual intersection size: 0
# Estimated intersection size (Theta): 0.00
# Actual difference size: 75000
# Estimated difference size (Theta): 74830.45


```
