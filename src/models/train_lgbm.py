import click
import joblib
import lightgbm
import pandas as pd


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_data_path', type=click.Path())
def main(input_data_path: str, output_data_path: str) -> None:

    data = pd.read_csv(input_data_path)
    X_train = data.drop(columns='result')
    y_train = data['result']

    model = lightgbm.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, output_data_path)


if __name__ == '__main__':
    main()
