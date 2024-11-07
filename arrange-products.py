import json
from collections import defaultdict, deque

# Load JSON data from a file
with open('products.json', 'r', encoding='utf-8') as f:
    products = json.load(f)

def arrange_products(products):
    # Step 1: Group products by supplier using a dictionary
    supplier_map = defaultdict(deque)
    for product in products:
        supplier_map[product['supplier']].append(product)

    # Step 2: Prepare to collect arranged products
    arranged_products = []

    # Step 3: Repeat selection round-robin style until all products are added
    while supplier_map:
        for supplier in list(supplier_map.keys()):
            # Get one product from the supplier's list
            arranged_products.append(supplier_map[supplier].popleft())

            # Remove the supplier from the map if no more products are left
            if not supplier_map[supplier]:
                del supplier_map[supplier]

    return arranged_products

# Arrange the products
arranged_products = arrange_products(products)

# Count unique suppliers
unique_suppliers = set(product['supplier'] for product in products)
num_unique_suppliers = len(unique_suppliers)

# Output the rearranged list and number of unique suppliers
with open('arranged_products.json', 'w') as file:
    json.dump(arranged_products, file, indent=2)

print("Products have been arranged with no consecutive suppliers.")
print("Number of unique suppliers:", num_unique_suppliers)
print("Unique Suppliers:")
for supplier in unique_suppliers:
    print(supplier)