import os
import dash
import us
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_table
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import json
import requests
import matplotlib.colors as mcolors
import matplotlib.cm as cm

#Set overlay colors for data

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True
if 'DYNO' in os.environ:
    app_name = os.environ['DASH_APP_NAME']
else:
    app_name = 'dash'


color_list = px.colors.qualitative.Alphabet

### Loading data and preprocessing
tsne_df = pd.read_csv('tsne_all.csv')

# Convert the people_ids column from string, to list of ints
tsne_df.loc[tsne_df['type'] == 'Event', 'people_ids'] = tsne_df.loc[tsne_df['type'] == 'Event', 'people_ids'] \
    .apply(lambda lst: list(map(lambda x: int(x), lst[1:-1].split(', '))))

# Get unique people, sorted by number of papers
people_names = tsne_df.sort_values('n_articles', ascending = False)['name'].unique().tolist()

# Mapping from person name -> person id
people_id_mapping = {name:int(idx) for name, idx in zip(tsne_df['name'], tsne_df['author_id']) if ~np.isnan(idx)}



### APP LAYOUT ###
app.layout = html.Div([
    html.Div([html.H1("Viz App for Workshop Recommender")],
             style={'textAlign': "center", "padding-bottom": "10", "padding-top": "10"}),

    dcc.Checklist(
        id = 'author_event_graph_checklist',
        options = [{'label': 'Show Authors (dot)', 'value': 'A'},
                   {'label': 'Show Events (square)', 'value': 'E'}],
        value = ['A', 'E']),

    dcc.Checklist(
        id = 'grey_out_checklist',
        options = [{'label': 'Grey out unselected?', 'value': 'Y'}],
        value = ['Y']),

    dcc.Checklist(
        id = 'select_all_people_checklist',
        options = [{'label': 'Select all people', 'value': 'Y'}],
        value = ['Y']),

    dcc.Dropdown(
        id = 'lda_dim_dropdown',
        options = [{'label': i, 'value': i} for i in range(20)],
        placeholder = 'Choose LDA Dimension',
        multi = True,
        value = list(range(20))),

    dcc.Dropdown(
        id = 'people_dropdown',
        options = [{'label': name, 'value': name} for name in people_names],
        placeholder = 'Choose Person',
        multi = True,
        value = []
    ),
        

    dcc.Graph(id = 'graph')
], className="container")



@app.callback(
    Output('graph', 'figure'),
    [Input('author_event_graph_checklist', 'value'),
    Input('lda_dim_dropdown', 'value'),
    Input('people_dropdown', 'value'),
    Input('grey_out_checklist', 'value'),
    Input('select_all_people_checklist', 'value')]
)
def draw_graph(checklist_values, dim_dropdown_values, people_dropdown_values, grey_out_checklist_value, select_all_people_checklist_value):

    tsne_all = tsne_df.copy()
    show_authors, show_events = 'A' in checklist_values, 'E' in checklist_values
    grey_out_checklist_value = 'Y' in grey_out_checklist_value
    select_all_people_checklist_value = 'Y' in select_all_people_checklist_value

    # Convert selected people names to their ids

    if  select_all_people_checklist_value == True:
        # Select all people
        selected_people_ids = list(set(people_id_mapping.values()))
    else:
        selected_people_ids = list(map(lambda x: people_id_mapping[x], people_dropdown_values))
    
    tsne_all['is_shown'] = False

    # Get columns of selected AUTHORS
    selected_people_cols = tsne_all['people_ids'].apply(lambda x: len(set(selected_people_ids).intersection(set(x))) > 0 if type(x) is list else False)

    # Get the columns of AUTHORS and EVENTS of selected people
    tsne_all.loc[selected_people_cols | tsne_all['author_id'].apply(lambda x: x in selected_people_ids), 'is_shown'] = True

    #Filter by max dimension 
    tsne_all['is_shown'] = (tsne_all['max_dim'].apply(lambda x: x in dim_dropdown_values) & tsne_all['is_shown'])

    # Filter by author shown or event shown
    if show_authors == False:
        tsne_all['is_shown'] = ~(tsne_all['type'] == 'Author') & tsne_all['is_shown']
    if show_events == False:
        tsne_all['is_shown'] = ~(tsne_all['type'] == 'Event') & tsne_all['is_shown']

    # After this, everything to be shown is True

    # If grey out , then filter and keep only those with is_shown = True
    if grey_out_checklist_value == False:
        tsne_all = tsne_all.loc[tsne_all['is_shown'] == True, :]
    
    fig = go.Figure()

    for i in range(20):

        if grey_out_checklist_value == True:

            tsne_subset = tsne_all.loc[(tsne_all['max_dim'] == i) & (tsne_all['type'] == 'Author') & (tsne_all['is_shown'] == False), :]
            # Add the authors hidden
            fig.add_trace(
            go.Scatter(
                x = tsne_subset['tsne_0'], 
                y = tsne_subset['tsne_1'], 
                mode = 'markers',
                marker_symbol = 0,
                opacity = 0.25,
                name = i,
                hovertext = tsne_subset['hover_text'], 
                marker = dict(
                    color = color_list[i],
                    size = 7,
                    line=dict(
                        color='black',
                        width=0.3
                    )
                ),
                )
            )

            tsne_subset = tsne_all.loc[(tsne_all['max_dim'] == i) & (tsne_all['type'] == 'Event') & (tsne_all['is_shown'] == False), :]
            # Add the events hidden
            fig.add_trace(
            go.Scatter(
                x = tsne_subset['tsne_0'], 
                y = tsne_subset['tsne_1'], 
                mode = 'markers',
                marker_symbol = 2,
                opacity = 0.25,
                name = i,
                hovertext = tsne_subset['hover_text'], 
                marker = dict(
                    color = color_list[i],
                    size = 7,
                    line=dict(
                        color='black',
                        width=0.3
                    )
                ),
                )
            )

        # Add the authors shown
        tsne_subset = tsne_all.loc[(tsne_all['max_dim'] == i) & (tsne_all['type'] == 'Author') & (tsne_all['is_shown'] == True), :]
        fig.add_trace(
        go.Scatter(
            x = tsne_subset['tsne_0'], 
            y = tsne_subset['tsne_1'], 
            mode = 'markers',
            marker_symbol = 0,
            opacity = 1,
            name = i,
            hovertext = tsne_subset['hover_text'], 
            marker = dict(
                color = color_list[i],
                size = 7,
                line=dict(
                    color='black',
                    width=0.3
                )
            ),
            showlegend = True
            )
        )

        # Add the events shown
        tsne_subset = tsne_all.loc[(tsne_all['max_dim'] == i) & (tsne_all['type'] == 'Event') & (tsne_all['is_shown'] == True), :]
        # Add the authors shown
        fig.add_trace(
        go.Scatter(
            x = tsne_subset['tsne_0'], 
            y = tsne_subset['tsne_1'], 
            mode = 'markers',
            marker_symbol = 2,
            opacity = 1,
            name = i,
            hovertext = tsne_subset['hover_text'], 
            marker = dict(
                color = color_list[i],
                size = 7,
                line=dict(
                    color='black',
                    width=0.3
                )
            ),
            showlegend = True
            )
        )
    

                                    
    fig.update_layout(
        width=1000,
        height=800,
        margin = {'l': 50, 'r': 50, 't': 50, 'b': 50},
        xaxis = dict(range = [-15, 15]),
        yaxis = dict(range = [-15, 15])
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=False)