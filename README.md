## Forecasting Sales Demand Using Amazon Forecast


[Wikipedia](https://en.wikipedia.org/wiki/Forecasting) defines Forecasting as the process of making predictions of the future based on past and present data and most commonly by analysis of trends. History is filled with humans making predictions by looking at trends and drawing conclusions from patterns. 

For businesses, the ability to forecast the future and make  informed business decisions is critical to their survival.  
Many organizations' traditional methods of generating forecasts from historical data often struggle to generate an accurate prediction from large data and data with irregular trends. 

For eCommerce stores, the ability to predict demand accurately is a critical need.  They need to know how many inventory store units to have at hand to be at full stock for each product at a given time.  A low inventory level increases the risk of having a stock out, and a tool high inventory level increases the cost related to handling inventory.


Recent research shows that [43% of online businesses](https://www.veeqo.com/inventory-management) admitted that they have inventory problems. Retailers have constant pressure to meet market demand. This pressure is exacerbated by the fact that consumers have many options. A customer with unmet demands will not likely return to the same retailer. 



Being able to predict an accurate demand is not limited to small businesses alone. In 2013, Walmart experienced[ out-of-stock across all stores](https://www.rsrresearch.com/research/the-walmart-out-of-stock-problem-lessons-learned). 



The big question stands: can recent advances in machine learning help retailers forecast demand? ML tools' democratization is empowering individuals with little to no ML skills to solve complex problems that previously required strong expertise in ML. AWS Forecast is one of these tools.


In this [notebook](./Sales_demand_forecast.ipynb), you'll learn how to predict sales demand using AWS Forecast. It's important to know that the workflow demonstrated in this notebook from data preprocessing to training a predictor can be fully automated.

- Demand forecast for product `moveis_decoracao` from September 1st to September 17th
![](https://res.cloudinary.com/samueljames/image/upload/v1602065910/Screenshot_2020-10-07_at_12.18.08.png)

- Demand forecast for product `pet_shop` from September 1st to September 17th

![](https://res.cloudinary.com/samueljames/image/upload/v1602312339/Screenshot_2020-10-10_at_08.45.07.png)

### Notebook
The notebook used for this tutorial can be found [here](./Sales_demand_forecast.ipynb)
