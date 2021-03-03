# workshop_recommender

Link to Visualization Tool: http://thamsuppp.pythonanywhere.com/

# Visualization Tool Instructions

Last updated 3/2/21

**Data**
The current data upon which the visualization was conducted:
- 932 authors who have published at least 3 papers on NIPS (paper text obtained from Kaggle dataset, LDA trained on first 20% of the paper text)
- 2156 events (workshops, talks, tutorials) in NIPS conferences from 2017 to 2019 (LDA trained on the event description, as scraped from NIPS Conference Schedule website)
- Authors and events are filtered so that every author here appears in at least one event here, and every event here has at least one author here.

**Basic Functionality**

The scatter plot shows the 2-dimensional t-SNE output of the 20-dimensional LDA vectors of events (square) and authors (dot).

The points are colored according to the dimension of maximum value in the LDA - hence allowing an intuitive check that points heavy in similar topics are placed closer to each other. 

You can toggle the checkboxes to display Authors and/or Events. The first dropdown menu allows you to subset authors and events by the max LDA dimension. The second dropdown menu allows you to select specific authors to visualize them (their author point and all the events they have spoken in). Accordingly, you have the option of either greying out unselected points or eliminating them entirely from the plot.

If select all people is checked, then all people in the dataset will be visualized, regardless of the value in the 'Choose person' dropdown.

Click on any point (author or event) and some basic information will be displayed. From there you can search the person/event title on Google with one click.

Note: Ignore the Test Code input box and button - that is for testing purposes.

**Speaker Recommendations**

Click on any event, and after its information is displayed, click the 'Generate Recommendations' button. Using the weights from the trained logistic regression model, every author will be given a recommendation score and the top 10 recommended authors will be returned in the table. Check on 'Show Recommendations' and these top 10 recommended authors will be displayed as red stars on the plot. (This allows you to visualize how similar in content are the recommended authors to the event in LDA space).

Current Recommendation algorithm used:
- Logistic regression model trained on 20 features: the absolute difference between the event and author among each of the 20 LDA dimensions. The class imbalance ratio was set to be 100:1, with **randomized** downsampling of the uninvited people.
Since there was an average of 1.65 speakers per event, there is a 167:1 ratio of samples to events. This model gave AUC of 0.63, and a recall @10 of 0.1988, with an average rank of actual speakers in model recommendation of 61.88 (out of ~166).

- (The naive univariate model gave a AUC of 0.62 recall@10 of 0.1833, with an average rank of actual speakers in model recommendation of 62.63 (out of ~166).


