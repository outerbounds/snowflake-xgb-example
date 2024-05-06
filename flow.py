from metaflow import (
    FlowSpec,
    step,
    kubernetes,
    Parameter,
    conda_base,
    conda,
    secrets,
    current,
    card,
    S3
)
from metaflow.cards import VegaChart


@conda_base(python="3.12")
class SnowflakeMLTrainingFlow(FlowSpec):

    n = Parameter("n", default=1_000_000, help="Number of rows to fetch from Snowflake")

    feature_db = Parameter(
        # snowflake_sample_data is a public database accessible in all Snowflake accounts.
        "fdb", default="snowflake_sample_data", help="Snowflake database name"
    )
    feature_schema = Parameter(
        # https://docs.snowflake.com/en/user-guide/sample-data-tpch
        "fschema", default="tpch_sf1", help="Snowflake feature schema name"
    )
    prediction_db = Parameter(
        # outerbounds_demo is a database you can make in your account using setup-preds-db.sql.
        "pdb", default="outerbounds_demo", help="Snowflake database name"
    )
    prediction_schema = Parameter(
        "pschema", default="tpch_predictions_schema", help="Snowflake prediction schema name"
    )
    prediction_table = Parameter(
        "ptable", default="lineitem_predictions", help="Snowflake prediction table name"
    )

    feature_columns = ["l_partkey", "l_suppkey", "l_quantity", "l_discount", "l_tax"]
    prediction_column = "l_extendedprice"
    model_file = 'model.json'

    @step
    def start(self):
        self.all_columns = self.feature_columns + [self.prediction_column]
        self.next(self.train)

    @card(type="blank", refresh_interval=1)
    @secrets(sources=[...]) # TODO: PUT YOUR SNOWFLAKE CREDS SECRET HERE
    @conda(
        libraries={
            "snowflake-connector-python": "3.7.1",
            "snowflake-sqlalchemy": "1.5.1",
            "pandas": "2.1.4",
            "scikit-learn": "1.4.2",
            "xgboost": "2.0.3",
            "altair": "5.3.0",
        }
    )
    @kubernetes
    @step
    def train(self):
        import os
        from snowflake.sqlalchemy import URL
        from sqlalchemy import create_engine
        import pandas as pd
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        import altair as alt

        ### Fetch data from Snowflake.
        engine = create_engine(
            URL(
                user=os.environ["SNOWFLAKE_USER"],
                password=os.environ["SNOWFLAKE_PASSWORD"],
                account=os.environ["SNOWFLAKE_ACCOUNT_IDENTIFIER"],
                warehouse="COMPUTE_WH",
                database=self.feature_db,
                schema=self.feature_schema,
            )
        )
        all_data = pd.read_sql(
            f'select {", ".join(self.all_columns)} from lineitem limit {self.n}', 
            con=engine
        )
        # NOTE: pandas changed the default behavior of read_sql connection in 2.2.0
        # https://stackoverflow.com/a/77949093

        ### Split data into train, validation, and holdout sets.
        # Holdout set will be used for unbiased evaluation of the model.
        X_train, self.X_holdout, y_train, self.y_holdout = train_test_split(
            all_data[self.feature_columns], all_data[self.prediction_column], test_size=0.05
        )
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train, y_train, test_size=0.2
        )

        ### Train the model.
        _card_rmse_data = []
        source = alt.Data({"values": _card_rmse_data})
        alt_chart = alt.Chart(source).mark_line(point=True).encode(x="epoch:Q", y="rmse:Q")
        chart = VegaChart.from_altair_chart(alt_chart)
        current.card.append(chart)

        class ProgressCallback(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                _card_rmse_data.append(
                    {"epoch": epoch, "rmse": evals_log["validation_0"]["rmse"][-1]}
                )
                chart.update(alt_chart.to_dict())
                current.card.refresh()

        model = xgb.XGBRegressor(
            eval_metric="rmse", callbacks=[ProgressCallback()]
        )
        model.fit(
            X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=0
        )
        model.save_model(self.model_file)
        with S3(run=self) as s3:
            s3.put_files([(self.model_file, self.model_file)])

        self.next(self.eval)

    @secrets(sources=[...]) # TODO: PUT YOUR SNOWFLAKE CREDS SECRET HERE
    @conda(
        libraries={
            "snowflake-connector-python": "3.7.1",
            "snowflake-sqlalchemy": "1.5.1",
            "pandas": "2.1.4",
            "scikit-learn": "1.4.2",
            "xgboost": "2.0.3"
        }
    )
    @kubernetes
    @step
    def eval(self):
        import os
        import xgboost as xgb
        from sklearn.metrics import root_mean_squared_error
        from snowflake.sqlalchemy import URL
        from sqlalchemy import create_engine
        import pandas as pd

        with S3(run=self) as s3:
            obj = s3.get(self.model_file)
            model = xgb.XGBRegressor(eval_metric="rmse")
            model.load_model(obj.path)

        y_hat = model.predict(self.X_holdout)

        # Option 1: score predictions
        self.rmse = root_mean_squared_error(y_hat, self.y_holdout.values)
        print(f"RMSE on holdout set: {self.rmse}")

        # Option 2: put predictions in Snowflake
        engine = create_engine(
            URL(
                user=os.environ["SNOWFLAKE_USER"],
                password=os.environ["SNOWFLAKE_PASSWORD"],
                account=os.environ["SNOWFLAKE_ACCOUNT_IDENTIFIER"],
                warehouse="COMPUTE_WH",
                database=self.prediction_db,
                schema=self.prediction_schema,
            )
        )
        preds_data = {"prediction": y_hat, 'id': range(len(y_hat))}
        pd.DataFrame(preds_data).to_sql(
            self.prediction_table, 
            con=engine, 
            if_exists="replace",
            index=False # Avoid `NotImplementedError: Snowflake does not support indexes`
        )

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SnowflakeMLTrainingFlow()
