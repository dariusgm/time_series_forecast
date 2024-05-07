import datetime
import json
import plotly.express as px
import pandas as pd
def main():
    # read excel file
    xls = pd.read_excel('data/Eltern-Kind Turnen Freitag 16-17.xlsx', sheet_name=None)
    # xls is a dictionary where the keys are the sheet names and the values are the dataframes
    participation = {}
    all_dates = set()
    all_families = set()
    for sheet_name, df in xls.items():
        print(f"Processing sheet: {sheet_name}")
        # now you can process each dataframe (df)
        for index, row in df.iterrows():
            if index < 2:
                continue

            first_name = row['Vorname/Kind']
            last_name = row['Name']
            full_name = f"{first_name} {last_name}"
            all_families.add(full_name)
            for col in df.columns:
                if type(col) is datetime.datetime:
                    key = f"{col.year}-{str(col.month).zfill(2)}-{str(col.day).zfill(2)}"

                    if row[col] == 1.0:
                        encoded_value = 1
                    else:
                        encoded_value = 0

                    if key in participation:
                        participation[key][full_name] = encoded_value
                    else:
                        participation[key] = {full_name: encoded_value}



    print(json.dumps(participation, indent=4))
    df = pd.DataFrame(participation).T


    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)

    # Fill missing participants that later joined with "0"
    df = df.fillna(0)

    # Drop the families where each date is "0"
    df = df.loc[:, ~(df == 0).all(axis=0)]

    # Sort the DataFrame by its index (the dates)
    df = df.sort_index()

    # Create a heatmap using Plotly Express
    fig = px.imshow(df, labels=dict(y="Date", x="Family", color="Participation"), title="Participation Heatmap")
    fig.show()



if __name__ == '__main__':
    main()