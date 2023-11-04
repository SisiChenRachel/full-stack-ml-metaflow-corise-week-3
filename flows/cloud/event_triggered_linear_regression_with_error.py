from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger,retry, catch
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@trigger(events=["s3"])
@conda_base(
    libraries={
        "pandas": "2.1.2",  
        "pyarrow": "13.0.0", 
        "scikit-learn": "1.3.2"
    }
)
class TaxiFarePrediction(FlowSpec):
    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):
        # TODO:
        # Try to complete tasks 2 and 3 with this function doing nothing like it currently is.
        # Understand what is happening.
        # Revisit task 1 and think about what might go in this function.

        obviously_bad_data_filters = [
            df.fare_amount > 0,  # fare_amount in US Dollars
            df.trip_distance <= 100,  # trip_distance in miles
            df.trip_distance > 0,
            # TODO: add some logic to filter out what you decide is bad data!
            # TIP: Don't spend too much time on this step for this project though, it practice it is a never-ending process.
            df.total_amount>0,
            df.passenger_count.notna()]

        for f in obviously_bad_data_filters:
            df = df[f]
        return df

  
    @step
    def start(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        self.df = self.transform_features(pd.read_parquet(self.data_url))

        # NOTE: we are split into training and validation set in the validation step which uses cross_val_score.
        # This is a simple/naive way to do this, and is meant to keep this example simple, to focus learning on deploying Metaflow flows.
        # In practice, you want split time series data in more sophisticated ways and run backtests.
        n = self.df.shape[0]
        self.params = [5, 100, n]
        
        self.next(self.linear_model_compute, foreach='params')


    @card(type="corise")
    @catch(var='compute_failed')
    @step
    def linear_model_compute(self):
        "Fit a single variable, linear model to the data."
        from sklearn.linear_model import LinearRegression

        # TODO: Play around with the model if you are feeling it.
        self.model = LinearRegression(copy_X=False,
                                        fit_intercept=False,
                                        n_jobs=-1,
                                        positive=True)

                            

        self.y = self.df["total_amount"].values
        self.sample_size = self.input
        self.X = self.df["trip_distance"].head(self.sample_size).values.reshape(-1, 1)


        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)
        current.card.append(Markdown("Accuracy: %0.2f (+/- %0.2f)" % (self.scores.mean(), self.scores.std() * 2)))
        
        self.next(self.join)

    @step
    def join(self, inputs):
        for input in inputs:
            if input.compute_failed:
                 print('compute failed for parameter: %d' % input.sample_size)
        self.next(self.end)

    @step
    def end(self):
        print("Success!")

if __name__ == "__main__":
    TaxiFarePrediction()

