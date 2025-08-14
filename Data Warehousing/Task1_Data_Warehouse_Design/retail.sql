-- Fact Table: Sales_Fact
CREATE TABLE Sales_Fact (
    sale_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    time_id INTEGER,
    store_id INTEGER,
    sales_amount DECIMAL(10, 2),
    quantity_sold INTEGER,
    FOREIGN KEY (customer_id) REFERENCES Customer_Dim(customer_id),
    FOREIGN KEY (product_id) REFERENCES Product_Dim(product_id),
    FOREIGN KEY (time_id) REFERENCES Time_Dim(time_id),
    FOREIGN KEY (store_id) REFERENCES Store_Dim(store_id)
);

-- Dimension Table: Customer_Dim
CREATE TABLE Customer_Dim (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    gender TEXT,
    age INTEGER,
    city TEXT,
    state TEXT,
    country TEXT
);

-- Dimension Table: Product_Dim
CREATE TABLE Product_Dim (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    brand TEXT,
    unit_price DECIMAL(10, 2)
);

-- Dimension Table: Time_Dim
CREATE TABLE Time_Dim (
    time_id INTEGER PRIMARY KEY,
    sale_date DATE,
    day_of_week TEXT,
    month TEXT,
    quarter INTEGER,
    year INTEGER
);

-- Dimension Table: Store_Dim
CREATE TABLE Store_Dim (
    store_id INTEGER PRIMARY KEY,
    store_name TEXT,
    city TEXT,
    state TEXT,
    country TEXT
);