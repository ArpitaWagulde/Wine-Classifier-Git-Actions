Github actions is setup such that it will run two pipelines.

First it runs the testing pipeline that will ensure that some of the sample inputs give correct outputs and that the accuracy of the model is above the set threshold.

Second it trains and saves the joblib file with the timestamp, allowing a version control.