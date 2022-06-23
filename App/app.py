import os
import sys
import shutil

import dash
import dash_bootstrap_components as dbc

from dash import html,dcc
from dash.dependencies import Input, State, Output

sys.path.append('./src')
from baseSetup import VCStorage
from utils import *

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "4rem 1rem 2rem",
    "background-color": "black",
}

CONTENT_STYLE = {
    "background-color": "black",
    'background-image':'url(assets/bk.png)',
    'background-repeat': 'no-repeat',
    'verticalAlign':'middle',
    'textAlign': 'center',
    'position':'absolute',
    'width':'100%',
    'height':'300%',
}

sidebar = html.Div(
    [
        dbc.Nav(
            [   
                html.Br(),
                html.Br(),
                html.H4("Select modality:", style={'textAlign':'center','color':'antiquewhite'},className="display-4"),
                
                dbc.NavLink("Audio Retrieval",style={'textAlign':'center','color':'antiquewhite'}, href="/audio", active="exact"),
                dbc.NavLink("Video Retrieval",style={'textAlign':'center','color':'antiquewhite'}, href="/video", active="exact"),
                dbc.NavLink("Multimodal Retrieval",style={'textAlign':'center','color':'antiquewhite'}, href="/multimodal", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

#modalities = ["Audio-based retrieval","Video-based retrieval","Multimodal Retrieval"]
vcBase = VCStorage()

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    content,
    sidebar
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    if os.path.isdir("tempKF"):
        shutil.rmtree("tempkF")
    
    if pathname == "/audio":
        return [
            html.H1("Music Video Retrieval Application",
                style={"textAlign":"centre","color":"antiquewhite","padding-top":"5%"}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div([
                    dbc.Row([
                        dbc.Col([dbc.Input(type="url",id="video-link",placeholder="Enter youtube link",debounce = True)],width=3),
                        dbc.Col([dbc.Button("Retrieve",id="submit-button", color="success",n_clicks=0,className="me-2")],width=1),
                            ],justify="center")
                    ]),
            html.Div(id="output_div")
                ]
    elif pathname == "/video":
        return [
            html.H1("Music Video Retrieval Application",
                    style={"textAlign":"centre","color":"antiquewhite","padding-top":"5%"}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div([
                    dbc.Row([
                        dbc.Col([dbc.Input(type="url",id="video-link",placeholder="Enter youtube link",debounce = True)],width=3),
                        dbc.Col([dbc.Button("Retrieve",id="submit-button", color="success",n_clicks=0,className="me-2")],width=1),
                            ],justify="center")
            
            
            
                    ]),
            html.Div(id="output_div")
            ]
    elif pathname == "/multimodal":
        return [html.H1("Music Video Retrieval Application",
                    style={"textAlign":"centre","color":"antiquewhite","padding-top":"5%"}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div([
                    dbc.Row([
                        dbc.Col([dbc.Input(type="url",id="video-link",placeholder="Enter youtube link",debounce = True)],width=3),
                        dbc.Col([dbc.Button("Retrieve",id="submit-button", color="success",n_clicks=0,className="me-2")],width=1),
                            ],justify="center")
                    ]),
            html.Div(id="output_div")
        
            ]


@app.callback(
            Output('output_div', 'children'),
            [Input('submit-button', 'n_clicks'),Input("url", "pathname")],
            [State('video-link', 'value')],
                  )
def update_output(clicks, modality, input_value):
    if (clicks is not None) and (input_value is not None):

        embUrl = embedUrlFromLink(input_value)
        
        if modality == "/audio":
            retrieved = audioBasedRetrieval(input_value,vcBase)
        elif modality == "/video":
            retrieved = videoBasedRetrieval(input_value,vcBase)
        elif modality == "/multimodal":
            retrieved = multimodalRetrieval(input_value,vcBase)

        retList = list()

        for idx in retrieved:
            retList.append(
                   
                dbc.Col([
                    dbc.Col([html.Img(src="{}".format(vcBase.MVBaseInfoFile[str(idx)]["thUrl"]),style={"height": "224px", "width": "224px"})]),                    
                    dbc.Col([dbc.Button(vcBase.MVBaseInfoFile[str(idx)]["ytTitle"],id="link-centered", className="ml-auto",color="secondary",href=vcBase.MVBaseInfoFile[str(idx)]["ytLink"])]),
                    html.Div(),
                    html.Div()
                    ])
                   
                   
                   )

        return [
                html.Br(),
                html.Br(),
                dbc.Row(dbc.Col(html.Iframe(src="{}".format(embUrl),style={"height": "450px", "width": "500px"}))),
                html.Div(retList)      
            ]

if __name__=='__main__':
    app.run_server(port= 5001,debug=False)