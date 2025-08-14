-- Roll-up: Group total sales by Country and Quarter
SELECT 
    c.Country,
    t.Quarter,
    SUM(f.TotalSales) AS TotalSales
FROM SalesFact f
JOIN CustomerDim c ON f.CustomerKey = c.CustomerKey
JOIN TimeDim t ON f.TimeKey = t.TimeKey
GROUP BY c.Country, t.Quarter
ORDER BY c.Country, t.Quarter;



PRAGMA table_info(SalesFact);
PRAGMA table_info(CustomerDim);
PRAGMA table_info(TimeDim);
