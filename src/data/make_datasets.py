import click
import pandas as pd


@click.command()
@click.argument('action_data_path', type=click.Path(exists=True))
@click.argument('person_data_path', type=click.Path(exists=True))
@click.argument('output_data_path', type=click.Path())
def main(action_data_path: str, person_data_path: str, output_data_path: str) -> None:

    action_train = pd.read_csv(
        action_data_path,
        index_col=0,
        parse_dates=['date'],
        dtype={
            'action_type': 'category',
            'char_1': 'category',
            'char_2': 'category',
            'char_3': 'category',
            'char_4': 'category',
            'char_5': 'category',
            'char_6': 'category',
            'char_7': 'category',
            'char_8': 'category',
            'char_9': 'category',
            'char_10': 'category',
        },
    )

    person_data = pd.read_csv(
        person_data_path,
        index_col=0,
        parse_dates=['date'],
        dtype={
            'char_1': 'category',
            'char_2': 'category',
            'char_3': 'category',
            'char_4': 'category',
            'char_5': 'category',
            'char_6': 'category',
            'char_7': 'category',
            'char_8': 'category',
            'char_9': 'category',
        },
    )

    dataset = action_train.merge(person_data, how='left', on='person_id', suffixes=('_a', '_p'))
    dataset.drop(columns=['person_id', 'action_id', 'date_a', 'date_p', 'group_1'], inplace=True)
    dataset.to_parquet(output_data_path, index=False)


if __name__ == '__main__':
    main()
