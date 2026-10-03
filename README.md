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


--------------------------------------------------------------------------------



 Some key insights of the EDA of the Hotel booking 

 <img width="1172" height="609" alt="image" src="https://github.com/user-attachments/assets/053d8e1c-f282-4dcd-9e37-6ee5faaca9ea" />
 Bookings by hotel type: City Hotels account for the majority of bookings at 61.1%, while Resort Hotels make up 38.9%.   Bookings by market segment: Online Travel Agents (Online TA) are the dominant segment at 59.1%, followed by Offline TA/TO (15.9%) and Direct bookings (13.5%).   Bookings by distribution channel: Travel Agents/Tour Operators (TA/TO) are the primary channel, driving 79.1% of bookings, with Direct channels bringing in 14.8%.   Bookings by customer type: Most customers are classified as Transient (82.4%), with Transient-Party making up the next largest group at 13.4%.   Bookings by deposit type: An overwhelming majority of bookings are made with No Deposit (98.7%).   Bookings by meal plan: Bed & Breakfast (BB) is the most popular meal plan at 77.8%, while SC (11.3%) and HB (10.4%) make up smaller portions.   
 <img width="1176" height="375" alt="image" src="https://github.com/user-attachments/assets/8d20ad4b-9b86-4b16-ad0f-fec99be854be" />
 Top guest countries (top 15 + Other): Portugal (PRT) is the most frequent country of origin at 31.4%, followed by Great Britain (GBR) at 11.9%, France (FRA) at 10.1%, and an aggregated "Other" category at 9.6%.   Bookings by agent (top 10 + others): "Agent 9" handles the highest volume of bookings at 32.9%, followed by "Other Agents" at 19.6%, "Agent 240" at 14.9%, and bookings made with "No Agent" at 13.9%.   Bookings by reserved room type: Room type "A" is the most frequently reserved at 64.7%, with room type "D" in second at 19.9% and room type "E" at 6.9%.   

 <img width="858" height="442" alt="image" src="https://github.com/user-attachments/assets/fee61aca-6b70-4851-9519-9b0d2a0951bf" />
 Peak Months: August records the highest number of bookings, surpassing 11,000, while July is the second busiest month with approximately 10,000 bookings.   

Lowest Months: January has the lowest arrival volume at under 5,000 bookings, with November and December also showing lower volumes around the 5,000 mark.   

General Trend: Booking numbers generally trend upward from January to August (with a slight drop in June) before declining sharply in September and remaining lower throughout the fall and winter. 

<img width="1181" height="487" alt="image" src="https://github.com/user-attachments/assets/1f93d954-64b5-450a-8c9b-34bbcccdc0d6" />

* **Distribution of lead_time:** Right-skewed distribution with a concentration of short lead times; the **mean** is 80.0 days and the **median** is 49.0 days.


* **Distribution of adr (adr > 0):** Heavily clustered near lower values with extreme outliers stretching up to over 5,000; the **mean** is 108.6 and the **median** is 99.0.


* **Distribution of total_nights (log scale):** Plotted on a logarithmic count scale, showing a steep drop-off with a long tail extending up to ~70 nights; the **mean** is 3.6 nights and the **median** is 3.0 nights.


* **Distribution of total_guests:** The vast majority of bookings consist of 1 to 4 guests, with a **median** of 2.0 guests and rare extreme outliers stretching past 50.


* **Distribution of booking_changes (log scale):** Discrete counts displayed on a log scale, heavily concentrated at 0 changes with a **mean** of 0.3 and a **median** of 0.0.


* **Distribution of days_in_waiting_list (log scale):** Dominated by 0 days on the waiting list; the **mean** is 0.7 days and the **median** is 0.0 days.


* **Distribution of previous_cancellations (log scale):** Almost all guests have zero previous cancellations; the **mean** is 0.0 and the **median** is 0.0, with a tiny tail extending up to ~26 cancellations.


* **Distribution of total_of_special_requests:** Concentrated primarily between 0 and 2 requests; the **mean** is 0.7 requests and the **median** is 0.0 requests.


<img width="1177" height="473" alt="image" src="https://github.com/user-attachments/assets/aefcba9a-9f9b-4d96-b4ca-604c9892b9c3" />

* **Boxplot of lead_time:** The bulk of the data (interquartile range) falls roughly between 0 and 150 days, with a heavy concentration of continuous outliers stretching beyond 700 days.


* **Boxplot of adr:** The main distribution is tightly compressed near 0, with a few outliers and one extreme isolated outlier exceeding 5,000.


* **Boxplot of total_nights:** Most stays are clustered under 10 nights, while scattered outliers extend in a long tail up to 70 nights.


* **Boxplot of adults:** The median sits tightly at roughly 2 adults, with several distinct outliers scattered across the scale, reaching past 50.


* **Boxplot of children & Boxplot of babies:** Both variables are heavily concentrated at 0, with isolated outliers appearing at integer intervals up to 10.


* **Boxplot of booking_changes:** The interquartile range is squashed at 0, with discrete outlier points appearing at integer marks up to roughly 18.


* **Boxplot of days_in_waiting_list:** The primary data sits exactly at 0, followed by a dense, horizontal line of numerous outliers stretching all the way to nearly 400 days.

<img width="1191" height="314" alt="image" src="https://github.com/user-attachments/assets/89f4c3c4-12fc-4c5a-8631-1356f2f3655d" />


* **Target variable: is_canceled:** A pie chart showing the class distribution, with 72.5% of bookings labeled as "Not cancelled" and 27.5% labeled as "Cancelled".


* **log(1 + lead_time):** A histogram with a density curve showing the log-transformed lead time. The transformation shifted the skewness to -0.72 (from a raw skew of 1.43), displaying a massive spike at 0 and a broader peak between 4 and 6.


* **log(1 + adr):** A histogram with a density curve showing the log-transformed average daily rate. It exhibits a skew of -0.73, forming a fairly normal, bell-shaped distribution that peaks near 4.5
  

  
<img width="1780" height="484" alt="image" src="https://github.com/user-attachments/assets/8756bfb6-8e9d-452c-b52f-1f88b8fa580c" />

The image displays three bar charts analyzing cancellation rates across different guest characteristics, referenced against a dashed line indicating the overall average cancellation rate of 27.5%:

* **Cancellation rate by is_repeated_guest:** First-time guests (0) cancel at a rate of 28%, slightly above the overall average, whereas repeated guests (1) have a dramatically lower cancellation rate of just 8%.


* **Cancellation rate by total_of_special_requests:** There is a clear downward trend in cancellations as special requests increase. Bookings with zero special requests have the highest cancellation rate at 33%, steadily declining to 22% for one request, and bottoming out at 6% for five requests.


* **Cancellation rate by country_grouped:** Brazil (BRA) and Portugal (PRT) experience the highest cancellation rates at 36%, followed by Italy (ITA) at 35% and China (CHN) at 32%. The lowest recorded rates are from the "Unknown" category (6%), Austria (AUT, 18%), and the Netherlands (NLD, 19%).

<img width="1222" height="354" alt="image" src="https://github.com/user-attachments/assets/bdc83136-7efb-483f-aa2b-27b00fca33d5" />


The image displays three charts analyzing the relationship between Average Daily Rate (ADR in euros per night) and stay/guest variables:

* **Lead time vs ADR (sample of 6,000):** A scatter plot showing most bookings clustered between 0 and 200 days of lead time with ADRs predominantly between 50 and 200 euros. There is no strong linear correlation, though variability in ADR narrows as lead time extends past 300 days.


* **Length of stay vs ADR (sample):** A scatter plot of total nights versus ADR, showing high density between 1 and 7 nights. Shorter stays exhibit a wider range of pricing (up to ~400 euros), while longer stays (10+ nights) show a tapering off in both frequency and top-end ADR.


* **ADR by number of guests:** A set of boxplots showing a clear, positive upward trend between guest count and room price. The median ADR rises steadily from approximately 75 euros for 1 guest to over 200 euros for 5 guests, with substantial upper outliers for 1 to 4 guests.

<img width="1884" height="484" alt="image" src="https://github.com/user-attachments/assets/0ee1f285-c068-473c-937d-edbcd94bd9a2" />

The image displays three charts analyzing Average Daily Rate (ADR in euro / night) across months, customer types, and market segments:

* **Average ADR by arrival month:** A line plot tracking seasonal pricing trends between hotel types. City Hotels maintain a relatively stable mean ADR between ~85 and ~130 euros year-round, peaking in May. In contrast, Resort Hotels experience dramatic summer seasonality, surging to a sharp peak of nearly 190 euros in August before dropping to around 50 euros in winter (November and January).


* **ADR by customer type and hotel:** Grouped boxplots comparing ADR between Resort Hotels and City Hotels across customer segments. City Hotels generally hold higher median ADRs across all customer categories (Transient, Contract, Transient-Party, and Group). Transient customers exhibit the highest rates and greatest variance, with numerous high-end outliers exceeding 300 to 500 euros.


* **ADR by market segment:** Boxplots showing price distributions across booking sources. "Online TA" and "Direct" command the highest median rates (centered around 100–110 euros) and the largest spread of premium outliers reaching up to 500 euros. "Complementary" and "Unknown" exhibit the lowest rates near 0–20 euros, while "Aviation" sits tightly around 100 euros with minimal variance.

<img width="2128" height="884" alt="image" src="https://github.com/user-attachments/assets/f28d8e01-f063-4e4b-8cc5-41fa1278fbc8" />

The image "image_9b6ae6.png" displays two lower-triangle correlation heatmaps comparing the relationships among key booking features using both linear (Pearson) and monotonic/rank (Spearman) correlations:

* **Pearson correlation heatmap:** Measures linear relationships between continuous and binary variables.


* **Notable Strong Positive Correlations:** `has_kids` and `total_guests` ($r = 0.67$); `total_guests` and `adr` ($r = 0.47$); `is_repeated_guest` and `previous_bookings_not_canceled` ($r = 0.44$); and `total_nights` and `lead_time` ($r = 0.32$).


* **Target (`is_canceled`) Associations:** Shows moderate negative correlations with `room_mismatch` ($r = -0.21$) and `required_car_parking_spaces` ($r = -0.18$), alongside positive correlations with `lead_time` ($r = 0.18$) and `adr` ($r = 0.13$).




* **Spearman correlation heatmap:** Evaluates monotonic relationships, reducing sensitivity to outliers and skewed distributions.


* **Strongest Correlations:** A very high rank correlation between `is_repeated_guest` and `previous_bookings_not_canceled` ($\rho = 0.80$), followed by `has_kids` and `total_guests` ($\rho = 0.58$), and `total_nights` with `lead_time` ($\rho = 0.46$).


* **Target (`is_canceled`) Associations:** Stronger positive rank correlation with `lead_time` ($\rho = 0.23$), while negative associations remain steady for `room_mismatch` ($\rho = -0.21$) and `required_car_parking_spaces` ($\rho = -0.19$).

* <img width="2058" height="534" alt="image" src="https://github.com/user-attachments/assets/8823127c-9c16-4e75-a66b-96a7e8c642b4" />

The image  presents three heatmaps cross-analyzing cancellation rates and ADR across different categorical dimensions:

* **Cancellation rate (%): lead time x market segment:** Shows cancellation risk categorized by lead-time buckets and booking channels. Longer lead times consistently elevate cancellation rates, particularly for "Online TA" (surging from 9% at 0–7 days to 87% past 365 days) and "Direct" (rising to 60% past 365 days). "Unknown" records a 100% cancellation rate at the 0–7 day mark.


* **Cancellation rate (%): deposit type x hotel:** Compares cancellation frequencies between City and Resort Hotels across deposit policies. "Non Refund" bookings show extremely high cancellations across both (97% in City Hotel, 84% in Resort Hotel). In contrast, "No Deposit" hovers near typical baseline rates (29% vs. 23%), while "Refundable" deposits show a significant divergence between City Hotels (67%) and Resort Hotels (17%).


* **Mean ADR: hotel x arrival month:** Illustrates the monthly average daily rate (in euros) across properties. City Hotel pricing remains stable throughout the year, fluctuating moderately between 87 and 130 euros. Resort Hotel rates exhibit severe summer peak seasonality, climbing from 50–59 euros in winter months up to 158 euros in July and 189 euros in August.

<img width="1021" height="900" alt="image" src="https://github.com/user-attachments/assets/c7a3ddd5-9fb0-4d73-9086-1829634dc514" />

The image "image_9b6efd.png" shows a lower-triangle pairplot of four key numerical variables sampled across 2,500 bookings, colored by the booking outcome: **Not cancelled** (green) and **Cancelled** (red):

* **Variables Included:** `lead_time`, `adr` (average daily rate), `total_nights`, and `total_of_special_requests`.


* **Diagonal (KDE Distributions):**
* **lead_time:** Both distributions peak near 0, but cancelled bookings have a noticeably fatter right tail extending toward longer lead times.


* **adr:** Both classes share similar unimodal distributions centered around 100 euro/night, with overlapping tails up to ~400.


* **total_nights:** Both groups peak between 1 and 4 nights, sharply dropping off beyond 7 nights.


* **total_of_special_requests:** Shows multi-modal discrete peaks at 0, 1, 2, 3, and 4 requests. Bookings with 0 special requests exhibit a noticeably higher proportion of cancellations compared to those with 1 or more requests.




* **Off-Diagonal (Scatter Plots):**
* **lead_time vs. adr / total_nights:** Demonstrates that cancellations (red dots) are distributed throughout the parameter space, but appear more densely in regions with longer lead times (>150 days) across varying stay lengths and price points.


* **total_of_special_requests Interactions:** As seen across the bottom row, higher bands of special requests (2, 3, and 4) are heavily dominated by green points ("Not cancelled"), confirming that guests with multiple requests are far less likely to cancel.

<img width="775" height="488" alt="image" src="https://github.com/user-attachments/assets/9adcb758-967e-4557-8e42-4736bc5f5881" />


The image displays a horizontal bar chart showing the **Top 12 features for predicting cancellation** using a **Random Forest** model, ranked by their feature importance scores:

* **Top Predictor:** `lead_time` is the most influential feature by a clear margin, scoring around 0.122 in feature importance.


* **Key High-Impact Features:**
* `total_of_special_requests` ranks second (~0.101).


* `country_grouped_PRT` is third (~0.088).


* `adr` follows in fourth place (~0.064).




* **Moderate Impact Features:**
* `arrival_date_week_number` (~0.045) and `previous_cancellations` (~0.044).


* `arrival_date_day_of_month` (~0.040).


* `agent_grouped_Agent 9` (~0.037).




* **Lower-Tier Top Predictors:** `total_nights` (~0.034), `stays_in_week_nights` (~0.030), `market_segment_Online TA` (~0.028), and `deposit_type_Non Refund` (~0.019).







  





 
