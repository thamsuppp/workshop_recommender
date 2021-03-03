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
import matplotlib.pyplot as plt

import gensim
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models

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

# Loading files
authors = pd.read_csv('Data/authors_filtered_lda20vectors.csv')
authors = authors.drop(['Unnamed: 0', 'author_id'], axis = 1)
events = pd.read_csv('Data/events_filtered_lda20vectors.csv')
events = events.drop('Unnamed: 0', axis = 1)
events['people_ids'] = events['people_ids'].apply(lambda lst: list(map(lambda x: int(x), lst[1:-1].split(', '))))
with open('Data/events_people_dict.json', 'r') as fp:
    events_people_dict = json.load(fp)
events_people_dict = {int(k): v for k, v in events_people_dict.items()}
authors.columns = [str(i) for i in range(20)] + ['author_id', 'n_articles', 'name']
ldamodel = LdaModel.load('Data/LDAModel_intro20.model')
dictionary = corpora.Dictionary.load('Data/dictionary_intro20')
reg_coefs = np.load('Data/model_coefs.npy')

### Loading data and preprocessing
tsne_df = pd.read_csv('tsne_all.csv')

# Convert the people_ids column from string, to list of ints
tsne_df.loc[tsne_df['type'] == 'Event', 'people_ids'] = tsne_df.loc[tsne_df['type'] == 'Event', 'people_ids'] \
    .apply(lambda lst: list(map(lambda x: int(x), lst[1:-1].split(', '))))

# Get unique people, sorted by number of papers
people_names = tsne_df.sort_values('n_articles', ascending = False)['name'].unique().tolist()

# Mapping from person name -> person id and vice versa
people_id_mapping = {name:idx for idx, name in zip(authors['author_id'], authors['name'])}
id_people_mapping = {idx:name for idx, name in zip(authors['author_id'], authors['name'])}


### APP LAYOUT ###
app.layout = html.Div([
    html.Div([
        
        html.H1("Viz App for Workshop Recommender")],
             style={'textAlign': "center", "padding-bottom": "10", "padding-top": "10"}),

        html.A('Readme', id = 'link', href='http://google.com', target='_blank'),     

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
            value = []),
        html.Div([
            html.H4('Info on Clicked Point'),
            html.Div('', id = 'click_info_div'),
            html.Button(html.A('Search on Google', id = 'link', href='http://google.com', target='_blank')),
            ]),
        html.Button('Generate Recommendations', id = 'rec_button'),
        html.Div([

            html.H4('Recommendations'),
            dcc.Checklist(
            id = 'show_recs_checklist',
            options = [{'label': 'Show Recommendations', 'value': 'Y'}],
            value = ['Y']),

            html.Div('', id = 'rec_div'),

            dash_table.DataTable(
                id = 'rec_table',
                columns = [{'name': i, 'id': i} for i in ['name', 'author_id', 'score']],
                data = [],
            ),


            dcc.Input(id = 'test_code_input', placeholder = 'type code...'),
            html.Button('Test Code', id = 'test_code_button'),
            html.Div('', id = 'test_div')
        ]),

        dcc.Graph(id = 'graph')
]
, className="container")


@app.callback(
    [Output('link', 'href'),
    Output('click_info_div', 'children')],
    [Input('graph', 'clickData')]
)
def display_metadata_on_click(click_data):

    if click_data:

        idx = click_data['points'][0]['customdata']
        hover_text = click_data['points'][0]['hovertext']
        item = tsne_df.loc[tsne_df['hover_text'] == hover_text, :]
        search_term = click_data['points'][0]['hovertext'].split('\n')[0]
        
        return ('http://google.com/search?q={}'.format(search_term), '{} \n {}'.format(idx, click_data['points'][0]['hovertext']))
    
    else:
        return (None, None)

@app.callback(
    Output('test_div', 'children'),
    [Input('test_code_button', 'n_clicks')],
    [State('test_code_input', 'value')])
def print_df_button(click_data, test_code_input):

    if test_code_input:
        eval(test_code_input)
    
    return  ''


@app.callback(
    Output('rec_table', 'data'),
    [Input('rec_button', 'n_clicks')],
    [State('graph', 'clickData')]
)
def recommendation(rec_button, click_data):
    if click_data:
        print(click_data['points'][0])
        # Get the event_id
        event_id = click_data['points'][0]['customdata']
        # Get the 20D event vector
        event_vector = events.loc[events['event_id'] == event_id, [str(e) for e in range(20)]].to_numpy()[0]
        event_actual_people_ids = events.loc[events['event_id'] == event_id, 'people_ids'].item()
        # Get every author's 20D vector
        authors_mat = authors.iloc[:, 0:20].to_numpy()
        # Get the features i.e. absolute value of diff between the event and author dimension
        x_mat = abs(authors_mat - event_vector)
        # Get scores for every author by multipling
        authors_scores = x_mat @ reg_coefs.T

        authors_scores_df = pd.concat([authors[['name', 'n_articles', 'author_id']], pd.DataFrame(authors_scores)], axis = 1)
        authors_scores_df.columns = ['name', 'n_articles', 'author_id', 'score']
        authors_scores_df = authors_scores_df.sort_values('score', ascending = False)

        # Get the ordered list of recommended people to the event
        recs_order = authors_scores_df['author_id'].tolist()
        print('Recs: {}'.format(recs_order[:10]))
        recs_names = [id_people_mapping[x] for x in recs_order]
        print('Recs Names: {}'.format(recs_names[:10]))

        # Get the recommendation order of the actual recommended people
        actual_order = [recs_order.index(actual_id) for actual_id in event_actual_people_ids]
        print('Rec Order of Actual Speakers: {}'.format(actual_order))

        rec_table = authors_scores_df[['name', 'author_id', 'score']][0:10].to_dict('records')

        return rec_table
    else:
        return ''

@app.callback(
    Output('graph', 'figure'),
    [Input('author_event_graph_checklist', 'value'),
    Input('lda_dim_dropdown', 'value'),
    Input('people_dropdown', 'value'),
    Input('grey_out_checklist', 'value'),
    Input('show_recs_checklist', 'value'),
    Input('select_all_people_checklist', 'value'),
    Input('rec_table', 'data')]
)
def draw_graph(checklist_values, dim_dropdown_values, people_dropdown_values, grey_out_checklist_value, 
show_recs_checklist_value, select_all_people_checklist_value, rec_table):

    tsne_all = tsne_df.copy()
    show_authors, show_events = 'A' in checklist_values, 'E' in checklist_values
    grey_out_checklist_value = 'Y' in grey_out_checklist_value
    select_all_people_checklist_value = 'Y' in select_all_people_checklist_value
    show_recs_checklist_value = 'Y' in show_recs_checklist_value

    print('show_recs_checklist_value is')
    print(show_recs_checklist_value)

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

    fig = go.Figure()
    # Plot the recommended people
    if show_recs_checklist_value == True:
        print('test')

        # Get recommended author ids
        rec_author_ids = [e['author_id'] for e in rec_table]
        print(rec_author_ids)

        # Think of the actual solution to this (nan and int becomes float)
        tsne_rec_subset = tsne_all.loc[tsne_all['author_id'].apply(lambda x: ~np.isnan(x) and int(x) in rec_author_ids), :]
        print(tsne_rec_subset)
        fig.add_trace(
        go.Scatter(
            x = tsne_rec_subset['tsne_0'], 
            y = tsne_rec_subset['tsne_1'], 
            customdata = tsne_rec_subset['event_id'],
            mode = 'markers',
            marker_symbol = 17,
            opacity = 1,
            name = 'Recs',
            hovertext = tsne_rec_subset['hover_text'], 
            marker = dict(
                color = 'red',
                size = 12,
                line=dict(
                    width=0.3
                )
            ),
            showlegend = True
            )
        )


    # If grey out , then filter and keep only those with is_shown = True
    if grey_out_checklist_value == False:
        tsne_all = tsne_all.loc[tsne_all['is_shown'] == True, :]
    
    for i in range(20):

        if grey_out_checklist_value == True:

            tsne_subset = tsne_all.loc[(tsne_all['max_dim'] == i) & (tsne_all['type'] == 'Author') & (tsne_all['is_shown'] == False), :]
            # Add the authors hidden
            fig.add_trace(
            go.Scatter(
                x = tsne_subset['tsne_0'], 
                y = tsne_subset['tsne_1'], 
                customdata = tsne_subset['author_id'],
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
                customdata = tsne_subset['event_id'],
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
            customdata = tsne_subset['author_id'],
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
            customdata = tsne_subset['event_id'],
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