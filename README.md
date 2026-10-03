# EDA_OF_HOTEL_BOOKING
Jupyter notebook EDA of the Hotel Booking Demand dataset (Kaggle): removes duplicates and target leakage, analyses what drives cancellations and room rates, and tests ML readiness with a baseline model (ROC-AUC 0.83).  Website field (


Abstract

This project presents an exploratory data analysis of the Hotel Booking Demand dataset, which contains 119,390 reservations made at a city hotel and a resort hotel in Portugal between 2015 and 2017. The goal was to assess the quality of the raw data, understand customer booking behaviour, and identify the factors linked to booking cancellations and room prices. After cleaning, 87,212 reliable bookings remained, of which 27.5% were cancelled. The analysis shows that cancellations depend mainly on how far in advance a guest books and through which channel, while prices depend on the hotel, the season and the party size.

Problem statement

Cancellations are a major revenue risk for hotels, because a room cancelled late often stays empty. If a hotel can tell which bookings are likely to be cancelled and what drives its prices, it can plan overbooking, deposits and pricing more intelligently. This project explores real booking data to uncover those patterns and to check whether the data is suitable for building a predictive model later.

Data cleaning

The raw dataset had several quality problems. About 26.8% of the rows were exact duplicates, and they inflated the cancellation rate from 27.5% to 37.0%, so they were removed. The company and agent columns were mostly empty because no company or agent was involved, so their missingness was turned into the features has_company and has_agent instead of being filled in. Placeholder values such as "Undefined", inconsistent country codes (CN and CHN) and 167 impossible records, such as bookings with zero guests or a negative price, were also fixed. Outliers were judged one variable at a time using business logic. Only 17 rows with clear data-entry errors were removed, for example a room rate of €5,400 per night and a booking for 55 adults.

Data leakage

An important finding was that the column reservation_status matches the target is_canceled perfectly, because it records the outcome of the booking. If it were used as a feature, a model would appear almost perfectly accurate while learning nothing useful. It was therefore dropped along with reservation_status_date. Features such as room mismatch and parking requests were also excluded from the baseline model, because they are only recorded at check-in, after a booking has already been kept or cancelled.

Key findings

Cancellation risk increases steadily with lead time, from about 8% for bookings made within a week to about 41% for bookings made more than a year ahead. Online travel agency bookings are the riskiest, and for them the cancellation rate rises from 9% to 87% as lead time grows. In contrast, guests with special requests, repeat guests and corporate customers cancel much less. Repeat guests cancel only 7.7% of the time compared with 28.3% for first-time guests. Prices are strongly seasonal: the resort's average daily rate rises from about €50 in November to about €189 in August, while the city hotel stays much steadier throughout the year.

Conclusion

The analysis showed that the dataset is rich and useful, but that it needs careful cleaning before use. Duplicates, target leakage and hidden placeholder values would all have led to misleading results if they had been ignored. A baseline Random Forest trained only on booking-time information, with a split by date, reached a ROC-AUC of about 0.83, which confirms that the cleaned data contains real predictive signal. The cleaned dataset can be used in future work for a cancellation classifier or a price regression model. Further steps would include threshold tuning and monitoring for drift, since the cancellation rate rose from about 20% in 2015 to about 32% in 2017.
