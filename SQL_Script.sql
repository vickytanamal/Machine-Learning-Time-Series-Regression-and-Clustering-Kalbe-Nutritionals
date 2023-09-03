/*Fill NULL Values*/
update customer set "Marital Status" = 'Married'
where "Marital Status" is null;

/*Customer’s Average Age based on Marital Status*/
select  "Marital Status", round(AVG(age)) as avg_age
from public.customer
group by 1;

/*Customer’s Average Age based on Gender*/
select gender, round(avg(age)) as avg_age 
from public.customer 
group by 1;

/*Store with The Most Total Sales Quantity*/
select s.storename, count(*) as total_qty from "transaction" t 
left join store s 
on t.storeid  = s.storeid 
group by 1
order by 2 desc
limit 1;

/*Product with The Most Sold Total Amount*/
select p."Product Name", sum(t.totalamount) as total_amount from "transaction" t 
left join product p 
on t.productid = p.productid 
group by 1
order by 2 desc
limit 1;