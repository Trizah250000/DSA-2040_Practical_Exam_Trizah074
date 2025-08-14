-- Roll-up Query: Total sales by country and quarter
SELECT 
    C.Country, 
    T.Year, 
    ((T.Month - 1) / 3 + 1) AS Quarter, 
    SUM(S.TotalSales) AS TotalSales
FROM SalesFact S
JOIN CustomerDim C ON S.CustomerPK = C.CustomerPK
JOIN TimeDim T ON S.TimePK = T.TimePK
GROUP BY C.Country, T.Year, ((T.Month - 1) / 3 + 1)
ORDER BY T.Year, Quarter, TotalSales DESC;

-- Drill-down Query: Sales details for Thailand by month
SELECT 
    T.Year, 
    T.Month, 
    SUM(S.TotalSales) AS TotalSales
FROM SalesFact S
JOIN CustomerDim C ON S.CustomerPK = C.CustomerPK
JOIN TimeDim T ON S.TimePK = T.TimePK
WHERE C.Country = 'Thailand'
GROUP BY T.Year, T.Month
ORDER BY T.Year, T.Month;

-- Slice Query: Total sales for Electronics category
SELECT 
    SUM(S.TotalSales) AS TotalElectronicsSales
FROM SalesFact S
JOIN ProductDim P ON S.ProductPK = P.ProductPK
WHERE P.Category = 'Electronics';