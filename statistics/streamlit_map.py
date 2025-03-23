import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import os
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Political Bias by State",
    page_icon="🗳️",
    layout="wide"
)

# File path
DATA_FILE = "statistics/news_political_data.csv"

# Dictionary mapping state names to their abbreviations
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

# Cache data with TTL of 1 day (in seconds)
@st.cache_data(ttl=86400)
def load_data():
    """Load and process the data from CSV file"""
    # Check if file exists
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file not found: {DATA_FILE}")
        return None, None
    
    # Get file modification time for display
    mod_time = os.path.getmtime(DATA_FILE)
    mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
    
    # Load data
    try:
        df = pd.read_csv(DATA_FILE)
        return df, mod_time_str
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def aggregate_state_data(df):
    """Aggregate data by state to calculate average political leaning"""
    # Group by state and calculate mean political leaning and count
    state_avg = df.groupby('state')['political_leaning'].agg(['mean', 'count']).reset_index()
    state_avg.columns = ['state', 'avg_political_leaning', 'article_count']
    
    # Add state abbreviations for the map
    state_avg['state_code'] = state_avg['state'].map(STATE_ABBREV)
    
    return state_avg

def create_state_map(state_data):
    """Create a choropleth map of political leanings by state"""
    # Make a copy to avoid modifying the original dataframe
    map_data = state_data.copy()
    
    # Create map
    fig = px.choropleth(
        map_data,
        locations='state_code',  # Use state abbreviations
        locationmode="USA-states",
        color='avg_political_leaning',
        scope="usa",
        color_continuous_scale=px.colors.diverging.RdBu_r,  # Red-Blue scale
        range_color=[-1, 1],  # Set range for political leaning
        hover_data=['state', 'avg_political_leaning', 'article_count'],
        labels={
            'avg_political_leaning': 'Political Leaning',
            'article_count': 'Number of Articles',
            'state': 'State'
        }
    )
    
    fig.update_layout(
        title="Average Political Bias by State",
        height=600,
        coloraxis_colorbar=dict(
            title="Political Leaning",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Liberal (-1)", "-0.5", "Neutral", "0.5", "Conservative (1)"]
        )
    )
    
    return fig

def create_bar_chart(state_data):
    """Create a bar chart of states by political leaning"""
    chart = alt.Chart(state_data).mark_bar().encode(
        x=alt.X('avg_political_leaning:Q', title='Political Leaning'),
        y=alt.Y('state:N', sort=alt.EncodingSortField(field='avg_political_leaning', order='ascending'), title='State'),
        color=alt.Color('avg_political_leaning:Q', 
                      scale=alt.Scale(domain=[-1, 1], scheme='redblue'),
                      title='Political Leaning'),
        tooltip=['state:N', 'avg_political_leaning:Q', 'article_count:Q']
    ).properties(
        width=600,
        height=800,
        title='States Ranked by Political Bias'
    )
    
    return chart

def create_source_chart(df):
    """Create a chart showing bias by news source"""
    # Group by source
    source_data = df.groupby('source')['political_leaning'].agg(['mean', 'count']).reset_index()
    source_data.columns = ['source', 'avg_political_leaning', 'article_count']
    
    # Filter sources with at least 2 articles
    source_data = source_data[source_data['article_count'] >= 2]
    
    # Create chart
    chart = alt.Chart(source_data).mark_bar().encode(
        x=alt.X('avg_political_leaning:Q', title='Political Leaning'),
        y=alt.Y('source:N', sort=alt.EncodingSortField(field='avg_political_leaning', order='ascending'), title='News Source'),
        color=alt.Color('avg_political_leaning:Q', 
                      scale=alt.Scale(domain=[-1, 1], scheme='redblue'),
                      title='Political Leaning'),
        size=alt.Size('article_count:Q', title='Article Count'),
        tooltip=['source:N', 'avg_political_leaning:Q', 'article_count:Q']
    ).properties(
        width=600,
        height=500,
        title='News Sources by Political Bias'
    )
    
    return chart

def main():
    # App title
    st.title("Political Bias by State")
    st.write("Interactive visualization of political bias in news coverage across US states")
    
    # Load data
    data_load = load_data()
    
    # Better error handling for data loading
    if data_load[0] is None:
        st.error("Failed to load data. Please check if the CSV file exists and has the correct format.")
        st.info("Make sure your CSV file is named 'news_political_data.csv' and has the columns: state, title, url, source, political_leaning")
        return
    
    df, last_modified = data_load
    
    # Display data refresh information
    st.sidebar.header("Data Information")
    st.sidebar.write(f"Last data update: {last_modified}")
    st.sidebar.write("Data automatically refreshes once per day")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Filter by news source
    sources = sorted(df['source'].unique())
    selected_sources = st.sidebar.multiselect("Select News Sources", sources, default=sources)
    
    if selected_sources:
        df_filtered = df[df['source'].isin(selected_sources)]
    else:
        df_filtered = df
    
    # Filter by political leaning range
    min_bias, max_bias = float(df['political_leaning'].min()), float(df['political_leaning'].max())
    bias_range = st.sidebar.slider(
        "Political Leaning Range", 
        min_value=min_bias, 
        max_value=max_bias,
        value=(min_bias, max_bias)
    )
    
    df_filtered = df_filtered[
        (df_filtered['political_leaning'] >= bias_range[0]) & 
        (df_filtered['political_leaning'] <= bias_range[1])
    ]
    
    # Aggregate data by state
    state_data = aggregate_state_data(df_filtered)
    
    # Debug info for map
    with st.expander("Debug Map Data"):
        st.write("Map data preview (first 10 rows):")
        st.dataframe(state_data.head(10))
        
        # Check for missing state codes
        missing_codes = state_data[state_data['state_code'].isna()]['state'].tolist()
        if missing_codes:
            st.warning(f"Missing state codes for: {', '.join(missing_codes)}")
    
    # Display US map
    st.header("Political Bias Map")
    try:
        map_chart = create_state_map(state_data)
        st.plotly_chart(map_chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating map: {e}")
        st.exception(e)
    
    # Create two columns for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            st.header("States Ranked by Political Bias")
            bar_chart = create_bar_chart(state_data)
            st.altair_chart(bar_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating state bar chart: {e}")
    
    with col2:
        try:
            st.header("News Sources by Political Bias")
            source_chart = create_source_chart(df_filtered)
            st.altair_chart(source_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating source chart: {e}")
    
    # Articles by state
    st.header("Articles by State")
    selected_state = st.selectbox("Select a state to view articles", 
                                options=sorted(df['state'].unique()))
    
    state_articles = df[df['state'] == selected_state].sort_values('political_leaning')
    
    for _, article in state_articles.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(f"Source: {article['source']}")
        with col2:
            # Political leaning indicator
            leaning = float(article['political_leaning'])
            color = "blue" if leaning < -0.2 else "red" if leaning > 0.2 else "gray"
            st.markdown(f"""
            <div style="text-align: center; color: {color}; font-weight: bold;">
                {leaning}
            </div>
            <div style="background: linear-gradient(to right, blue, white, red); 
                height: 20px; border-radius: 5px; position: relative;">
                <div style="position: absolute; left: {(leaning + 1) * 50}%; 
                    transform: translateX(-50%); width: 5px; height: 20px; 
                    background-color: black;"></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
    
    # Raw data section
    with st.expander("View Raw Data"):
        st.dataframe(df_filtered)

if __name__ == "__main__":
    main()