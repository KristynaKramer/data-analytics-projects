import streamlit as st
import pandas as pd
import datetime as dt
import seaborn as sns
import numpy as np
import requests # for API calls
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from io import StringIO # for pandas to read csv received by API

#REFERENCE_START = 1961
#REFERENCE_END = 1990
START_DATE = '1940-01-01'
END_DATE = (dt.date.today() - dt.timedelta(days=2)).isoformat()
THIS_YEAR = dt.date.today().year
MIN_EXC = 5 # minimal temperature excess
MIN_DAYS = 5 # minimal number of consecutive days

FIGSIZE = (10, 8)
MAROON = '#550000'
BROWN = '#7A4300'
ORANGE = '#D07D18'
LIGHTBLUE = '#77A6CF'
BLUE = '#080043'
GRAY = '#86858A'



# Change font size for plots
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rcParams.update({'font.size': BIGGER_SIZE})

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



#---------------------------------------------------------
#---------------------------------------------------------
# --- FUNCTIONS ---

@st.cache_data
def search_city(city):
    '''Returns a list of matching results, or None if no matches.'''
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": city, "count": 5}
    
    response = requests.get(geo_url, geo_params)
    data = response.json()
    
    if "results" not in data or len(data["results"]) == 0:
        return None
    
    return data["results"]

#---------------------------------------------------------

def select_correct_city(list):
    options = []
    for r in list:
        region = r.get("admin1", "")
        country = r.get("country", "")
        label = ", ".join(filter(None, [r["name"], region, country]))
        options.append(label)

    choice = st.radio("Select the correct city:", options)

    if st.button("Confirm"):
        index = options.index(choice)
        st.session_state.selected_city = list[index]

#--------------------------------------------------------

def get_city_details():
    '''
    Returns (name, admin1, country, latitude, longitude) 
    from the currently selected city, or None if nothing selected yet.
    '''
    city = st.session_state.selected_city
    if city is None:
        return None
    return {
        "name": city["name"],
        "admin1": city.get("admin1", ""),
        "country": city.get("country", ""),
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        }

#---------------------------------------------------------
@st.cache_data
def load_data(latitude, longitude):
    '''
    Loads the data using API
    for the given city 
    from "open-meteo.com"
    '''

    # --- Pull data ---
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_max,temperature_2m_mean",
        "timezone": "auto",
        "format": "csv"
    }
    response = requests.get(url, params=params, timeout=60)

    # --- Load data into DataFrame ---
    df = pd.read_csv(StringIO(response.text), skiprows=3)

    return df

#---------------------------------------------------------
@st.cache_data
def prepare_data(lat, long, REFERENCE_START, REFERENCE_END):
    '''
    Preprocesses the data
    '''
    
    # --- Read data ---
    df = load_data(lat, long)

    # --- Prepare the dataset ---
    df.rename(
        columns={
            'temperature_2m_max (°C)' : 'max', 
            'temperature_2m_mean (°C)' : 'mean', 
            'time' : 'date'
            }, 
        inplace=True
        )
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # --- Delete data for February 29 ---
    df = df[(df['day'] != 29) | (df['month'] != 2)] 

    # --- Prepare the reference values ---
    reference_daily = df[
                        (df['year'] >= REFERENCE_START)
                        & (df['year'] <= REFERENCE_END)
                        ]
    reference = reference_daily.groupby(
        ['month','day']
        ).agg(
            {
                'max': ['mean', lambda x: x.std(ddof=0)], #plain std uses ddfo=1, so it divides by N-1,
                'mean': ['mean', lambda x: x.std(ddof=0)] #which does not work for reference period of length 1
            }
            ).reset_index()
    # change the two-level headings to one-level
    reference.columns = [
        '_'.join(filter(None, col)).strip() 
        if isinstance(col, tuple) 
        else col 
        for col in reference.columns
        ] 
    # rename the columns
    reference.rename(
        columns={
            'max_<lambda_0>' : 'max_std',
            'mean_<lambda_0>' : 'mean_std'
        },
        inplace=True
    )

    # --- Return ---
    return df, reference

#---------------------------------------------------------
@st.cache_data
def heatwave_compare_data(latitude, longitude, how, REFERENCE_START, REFERENCE_END):
    ''' 
    Prepares a dataset used for heatwave plotting
    Uses heatwave definition with the difference: 
            MIN_EXC (= 5°C) if how = 'fixed'
            std if how = 'std'
    '''

    daily, ref = prepare_data(latitude, longitude, REFERENCE_START, REFERENCE_END)
    compare = daily.merge(
        ref, 
        how='left', 
        on=['month','day'], 
        suffixes=('_year', '_ref')
        )
    compare.rename(
        columns={
            'max': 'year_max', # daily max
            'mean' : 'year_mean', # daily mean
            'max_mean' : 'ref_max_mean', # mean over reference period of daily max
            'max_std' : 'ref_max_std', # std over reference period of daily max
            'mean_mean' : 'ref_mean_mean', # mean over reference period of daily mean
            'mean_std' : 'ref_mean_std' # std over reference period of daily max
            }, 
        inplace=True
        )
    if how == 'fixed':
        compare['hot'] = compare['ref_max_mean'
                ] + MIN_EXC <= compare['year_max']
    elif how == 'std':
        compare['hot'] = compare['ref_max_mean'
                ] + compare['ref_max_std'] <= compare['year_max'] 
    else:  
        raise ValueError('how must be "fixed" or "std".')
        
    # Mark if the day is the end of a MIN_DAYS-long hot period    
    compare['heatwave_end'] = (
          compare['hot'].rolling(MIN_DAYS).min() == 1
          ) 
    # However, we miss days 1-4 of each heatwave
    # So check if at least one of the following MIN_DAYS is a heatwave_end
    # min_periods=1 makes it work at the end of the dataset
    compare['heatwave'] = (
          compare['heatwave_end'][::-1]
            .rolling(MIN_DAYS, min_periods=1)
            .max()[::-1] == 1
          )
    
    # Only store the temperature if the day was part of a heatwave
    compare['heatwave_temp'] = None
    compare.loc[
          compare['heatwave'], 'heatwave_temp'
          ] = compare['year_max']
    
    return compare

#---------------------------------------------------------
@st.cache_data
def heatwave_days_in_year(df, year):
    ''' 
    Counts the number of days in the given year that satisfy 
    the definition of a heatwave.
    '''
    
    df_year = df[df['year'] == year]
    return df_year['heatwave'].sum()

#---------------------------------------------------------

def heatwave_count_plot(latitude, longitude, how):
    '''
    Creates a barplot with number of heatwave days per year 
    for the given city.
    
    Uses heatwave definition with the difference: 
            MIN_EXC (= 5°C) if how = 'fixed'
            std if how = 'std'
    '''
    
    compare = heatwave_compare_data(latitude, longitude, how, REFERENCE_START, REFERENCE_END)

    heatwave_days = []
    for year in range(1941, THIS_YEAR+1):
        heatwave_days.append(
            heatwave_days_in_year(compare, year)
            )
    
    heatwave_days_reference = []
    for year in range(REFERENCE_START, REFERENCE_END+1):
        heatwave_days_reference.append(
            heatwave_days_in_year(compare, year)
            )


    heatwave_days_reference_mean = np.mean(
        heatwave_days_reference
        )

    fig = plt.figure(figsize=FIGSIZE)
    xaxis = range(1941, THIS_YEAR+1)
    ax = sns.barplot(x=xaxis, y=heatwave_days, color=MAROON)
    ax.axhline(
        y=heatwave_days_reference_mean, 
        color=BLUE, linestyle='--', 
        label = 'Average of {start} – {end}'.format(
            start=REFERENCE_START, end=REFERENCE_END
            )
        )
    
    if how == 'fixed':
        method = str(MIN_EXC)+'°C'
    else:
        method = 'std'
    ax.set_title(
        'Number of days in year that count as a heatwave '
        '({method})'.format(method=method),
        fontweight="bold", size=14, pad=10
        )
    ax.set_xticks(range(0, len(xaxis), 10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True, axis='y', color='gray', alpha=0.3)
    plt.legend()
    st.pyplot(fig)


    if how == 'fixed':
        method_long = str(MIN_EXC)+'°C'
    else:
        method_long = 'standard deviation'
    st.write(
        'The plot shows the number of days per year that classify as a heatwave. ' \
        'A heatwave occurs when the daily maximum temperature of more than five consecutive days exceeds the average maximum temperature by {method} compared to the selected reference period {start} – {end}.'.format(method=method_long, start=REFERENCE_START, end=REFERENCE_END)
    )

#---------------------------------------------------------

def heatwave_days_plot(latitude, longitude, how, years):
    '''
    Plots the temperatures during heatwaves for the given 
    city and given years (list).
    
    Uses heatwave definition with the difference: 
            MIN_EXC (= 5°C) if how = 'fixed'
            std if how = 'std'
    '''

    # --- load and prepare data ---
    daily, ref = prepare_data(latitude, longitude, REFERENCE_START, REFERENCE_END)
    compare = heatwave_compare_data(latitude, longitude, how, REFERENCE_START, REFERENCE_END)

    # --- PLOT ---
    fig = plt.figure(figsize=FIGSIZE)
    days = range(1,366)

    # --- X axis: ticks at 1st of each month ---
    month_starts = []
    month_labels = [''] * 13 # hide tick labels
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',''
        ]

    base = dt.date(2026, 1, 1)
    for m in range(1, 13):
        d = dt.date(2026, m, 1)
        day_of_year = (d - base).days
        month_starts.append(day_of_year)

    month_starts.append(365) # Add Dec 31 as final boundary

    # --- plot ---
    ax = plt.subplot()
    plt.plot(
        ref['max_mean'], 
        label='Average of {start} – {end}'.format(
            start=REFERENCE_START, end=REFERENCE_END
            ), 
        color=BLUE
        )
    
    if how == 'fixed':
        plt.fill_between( 
                    ref.index,
                    ref['max_mean'], 
                    ref['max_mean'] + MIN_EXC,
                    color=LIGHTBLUE,
                    alpha = 0.2,
                    label='{}°C above average'.format(MIN_EXC)
                    )
    
    elif how == 'std':
        plt.fill_between( 
                    ref.index,
                    ref['max_mean'], 
                    ref['max_mean'] + ref['max_std'],
                    color=BLUE,
                    alpha = 0.2,
                    label='std above average'.format(MIN_EXC)
                    )

    ax.set_prop_cycle(color=plt.cm.Dark2.colors)

    for year in years:
        if year == THIS_YEAR: 
            color = MAROON 
        elif year == THIS_YEAR-1:
            color = ORANGE
        else: 
            color = None

        compare_year = compare[compare['year']==year].reset_index()
        plt.plot(compare_year['heatwave_temp'], label=str(year), color=color)

    ax.set_xlim(1, 365)
    ax.set_xticks(month_starts)
    empty = '            '
    ax.set_xticklabels([empty + month for month in month_names])

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:.0f}°C')
        )

    ax.grid(True, color='gray', alpha=0.3)

    if how == 'fixed':
        method = str(MIN_EXC)+'°C'
    else:
        method = 'std'
    ax.set_title(
        'Daily maximum temperatures during heatwaves ({})'.format(method),
        fontweight="bold", size=14, pad=10
        )
    plt.legend()
    st.pyplot(fig)

    if how == 'fixed':
        method_long = str(MIN_EXC)+'°C'
    else:
        method_long = 'standard deviation'
    st.write(
        'The plot shows the daily maximum temperatures during heatwaves. ' \
        'A heatwave occurs when the daily maximum temperature of more than five consecutive days exceeds the average maximum temperature by {method} compared to the selected reference period {start} – {end}.'.format(method=method_long, start=REFERENCE_START, end=REFERENCE_END)
    )

#---------------------------------------------------------

def daily_max_plot(latitude, longitude, years):
    '''
    Plots the daily maximum temperatures for the last two years, 
    in comparison with the average of the reference period 
    and standard deviation.
    '''

    daily, ref = prepare_data(latitude, longitude, REFERENCE_START, REFERENCE_END)

    # --- PLOT ---
    fig = plt.figure(figsize=FIGSIZE)
    days = range(1,366)

    # --- X axis: ticks at 1st of each month ---
    month_starts = []
    month_labels = [''] * 13 # hide tick labels
    month_names = [
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',''
                    ]

    base = dt.date(2026, 1, 1)
    for m in range(1, 13):
        d = dt.date(2026, m, 1)
        day_of_year = (d - base).days
        month_starts.append(day_of_year)

    month_starts.append(365) # Add Dec 31 as final boundary

    # --- plot ---
    ax = plt.subplot()

    plt.plot(
        ref['max_mean'], 
        label='Average of {start} – {end}'.format(
            start=REFERENCE_START, end=REFERENCE_END
            ),
        color=BLUE
        )
    plt.fill_between( 
                ref.index,
                ref['max_mean'] - ref['max_std'], 
                ref['max_mean'] + ref['max_std'],
                color=BLUE,
                alpha = 0.2,
                label='std'
                )

    for year in years:

        if year == THIS_YEAR: 
            color = MAROON 
        elif year == THIS_YEAR-1:
            color = ORANGE
        else: 
            color = None

        plt.plot(
            daily[daily['year']==year].reset_index()['max'],
            label=year,
            color=color
        )

    ax.set_xlim(1, 365)
    ax.set_xticks(month_starts)
    empty = '            '
    ax.set_xticklabels([empty + month for month in month_names])

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:.0f}°C')
        )

    ax.grid(True, color='gray', alpha=0.3)

    ax.set_title(
        'Daily maximum temperatures',
        fontweight="bold", size=14, pad=10
        )
    plt.legend()
    st.pyplot(fig)

    st.write(
        'The graph shows the maximum daily temperatures of the selected years, ' \
        'and compares them with the average temperature and the standard deviation ' \
        'of the reference period {} – {}.'.format(REFERENCE_START, REFERENCE_END)
    )

#---------------------------------------------------------

def spaghetti_plot(latitude, longitude, years):
    '''
    Plots the Copernicus 'spaghetti' chart for daily maximum 
    temperatures and the current year.
    '''

    daily, ref = prepare_data(latitude, longitude, REFERENCE_START, REFERENCE_END)

    # --- PLOT ---
    fig = plt.figure(figsize=FIGSIZE)
    days = range(1,366)

    # --- X axis: ticks at 1st of each month ---
    month_starts = []
    month_labels = [''] * 13 # hide tick labels
    month_names = [
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',''
                    ]

    base = dt.date(2026, 1, 1)
    for m in range(1, 13):
        d = dt.date(2026, m, 1)
        day_of_year = (d - base).days
        month_starts.append(day_of_year)

    month_starts.append(365) # Add Dec 31 as final boundary

    # --- plot ---
    ax = plt.subplot()
    for year in range(1941, THIS_YEAR):
        data_year = daily[daily['year']==year].reset_index()
        plt.plot(data_year['max'], color=GRAY, linewidth=0.5)

    data_2025 = daily[daily['year']==THIS_YEAR-1].reset_index()
    plt.plot(
        data_2025['max'], 
        color=GRAY, 
        linewidth=0.5, 
        label='1941 – {}'.format(THIS_YEAR)
        )
        # this is a trick how to show one line legend 
        # for all the thin gray lines
    
    plt.plot(
        ref['max_mean'], 
        label='Average of {start} – {end}'.format(
            start=REFERENCE_START, end=REFERENCE_END
            ), 
        color=BLUE, 
        linestyle='--'
        )
    
    for year in years:
        if year == THIS_YEAR: 
            color = MAROON 
        elif year == THIS_YEAR-1:
            color = ORANGE
        else: 
            color = None

        data_year = daily[daily['year']==year].reset_index()
        plt.plot(
            data_year['max'], 
            label=year, 
            color=color, 
            linewidth=2
            )

    ax.set_xlim(1, 365)
    ax.set_xticks(month_starts)
    empty = '            '
    ax.set_xticklabels([empty + month for month in month_names])

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:.0f}°C')
        )

    ax.grid(True, color='gray', alpha=0.3)

    ax.set_title(
        'Daily maximum temperatures',
        fontweight="bold", size=14, pad=10
        )
    plt.legend()
    st.pyplot(fig)

    st.write(
        'This graph is nicknamed "Copernicus spaghetti plot". It shows ' \
        'the daily maximum temperatures for the entire period 1941 – {this_year} ' \
        'in thin grey lines, and the average maximum temperature of the reference ' \
        'period {start} – {end}. You can select the years for which the daily maximum ' \
        'temperatures are displayed in bold lines.'.format(this_year=THIS_YEAR, start=REFERENCE_START, end=REFERENCE_END)
    )

#---------------------------------------------------------

def select_years():
    '''
    Creates checkboxes for years.
    Returns list of checked years.
    Initial value is [THIS_YEAR-1, THIS_YEAR].
    '''
    st.write('Select years to be displayed:')     
    years = list(range(THIS_YEAR-19, THIS_YEAR+1))
    default_selected = {THIS_YEAR-1, THIS_YEAR}  # years checked by default

    n_cols = 5
    selected_years = []
    cols = st.columns(n_cols)
    for i, year in enumerate(years):
        col = cols[i % n_cols]
        checked = col.checkbox(
            str(year),
            value=(year in default_selected),
            key=f"year_{year}"
        )
        if checked:
            selected_years.append(year)

    return selected_years

#---------------------------------------------------------
# --- Warming stripes ---

def prepare_data_stripes(latitude, longitude):
    '''
    Loads the data for the given city from "open-meteo-city.csv"
    Prepares the data for use for drawing the warming stripes
    '''
    
    # --- Read data ---
    df = load_data(latitude, longitude)

    # --- Prepare the dataset ---
    df.rename(
        columns={
            'temperature_2m_mean (°C)' : 'temperature', 
            'time' : 'date'
            }, 
        inplace=True
        )
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df[
        (df['year'] < THIS_YEAR) & (df['year'] > 1940)
        ] # delete incomplete years 1940 and THIS_YEAR

    # --- Compute yearly mean temperatures ---
    yearly = df.groupby(['year'])['temperature'].mean().reset_index()

    # --- Compute the reference average temperature ---
    reference = yearly[
                        (yearly['year'] >= REFERENCE_START)
                        & (yearly['year'] <= REFERENCE_END)
                        ]
    reference_average = reference['temperature'].mean()

    # --- Compute anomalies ---
    yearly['anomalies'] = yearly['temperature'] - reference_average
    
    return yearly, reference_average

#---------------------------------------------------------

def warming_stripes_plot(latitude, longitude):
    '''
    Plots the warming stripes for the given city
    '''

    df, avg = prepare_data_stripes(latitude,longitude)

    # Create the figure and axes objects, specify the size 
    # and the dots per inches 
    fig, ax = plt.subplots(figsize=(10,3), dpi = 96)

    # Colours - Choose the colour map - 8 blues and 8 reds
    # the darkest blue is guessed, the darkest red is measured in Paint
    cmap = ListedColormap([
        "#07295d", '#08306b', '#08519c', '#2171b5', '#4292c6', 
        '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
        '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
        '#ef3b2c', '#cb181d', '#a50f15', '#67000d', '#440007'])

    # linearly normalizes data into the [0.0, 1.0] interval
    #norm = mpl.colors.Normalize(
    #        df['anomalies'].min(), 
    #        df['anomalies'].max()
    #        )

    # normalizes data 
    # boundary between blue and red is the average of the reference period
    #max_overall = df['temperature'].max()
    #min_overall = df['temperature'].min()
    #max_abs = max(np.abs(max_overall-avg), np.abs(min_overall-avg))
    #norm = mpl.colors.TwoSlopeNorm(
    #    vmin=-max_abs,
    #    vcenter=0,
    #    vmax=max_abs
    #)

    # another way how to normalize data
    # zero is still the reference average
    # but color scale is 
    baseline_period = df[(df['year'] >= 1901) & (df['year'] <= 2000)]
    std_1901_2000 = baseline_period['temperature'].std()

    max_abs = 3.0 * std_1901_2000

    norm = mpl.colors.TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs
    )

    # Plot bars
    bar = ax.bar(
        df['year'], 
        1, 
        color=cmap(norm(df['anomalies'])), 
        width=1, 
        zorder=2
        )

    # Remove the spines
    ax.spines[
        ['top', 'left', 'bottom', 'right']
        ].set_visible(False)

    # Reformat y-axis label and tick labels
    ax.set_ylabel('', fontsize=12, labelpad=10)
    ax.set_yticks([])
    ax.set_ylim([0, 1]) 

    # Adjust the margins around the plot area
    plt.subplots_adjust(
        left=0.1, 
        right=None, 
        top=None, 
        bottom=0.2, 
        wspace=None, 
        hspace=None
        )

    # Set a white background
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')

    # Reformat x-axis label and tick labels
    ax.set_xlabel('', fontsize=12, labelpad=10)
    ax.xaxis.set_tick_params(
        pad=2, 
        labelbottom=True, 
        bottom=True, 
        labelsize=12, 
        labelrotation=0, 
        color='white'
        )
    ax.set_xlim([df['year'].min(), df['year'].max()+1])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # Set graph title
    ax.set_title(
        'Temperature change '
        '(1941 - {year})'.format(year=THIS_YEAR-1), 
        #loc='left', 
        fontweight="bold", size=14, pad=10
        )

    # Adjust the margins around the plot area
    plt.subplots_adjust(
        left=0.11, 
        right=None, 
        top=None, 
        bottom=0.2, 
        wspace=None, 
        hspace=None
        )
    
    st.pyplot(fig)
    st.write(
        'The boundary between blue and red colors is set ' \
        'to be the average temperature of the reference period you chose, ' \
        'which is {} – {}. ' \
        'The color scale varies from +/- 3.0 standard deviations ' \
        'of the annual average temperatures of 1901 – 2000. ' \
        'This is consistent with the original [#ShowYourStripes](https://showyourstripes.info/).'.format(REFERENCE_START, REFERENCE_END)
        )


#---------------------------------------------------------
#---------------------------------------------------------
# --- STREAMLIT FLOW ---

st.title('Heatwaves around the world')

st.write('Start by searching for a city in the sidebar.')

# Reference period slider on the sidebar
with st.sidebar:
    (REFERENCE_START, REFERENCE_END) = st.slider(
        label='Reference period:', 
        min_value=1941,
        max_value=THIS_YEAR-1,
        value=(1961,1990),
        step=1, 
        )

st.write('Data are pulled from [open-meteo.com](https://open-meteo.com/en/docs/historical-weather-api) '
    'for the period {} to {}.'.format(START_DATE,END_DATE))


# Initialize state on first run
if "results" not in st.session_state:
    st.session_state.results = None
if "selected_city" not in st.session_state:
    st.session_state.selected_city = None

# --- Stage 1: search for a city ---
with st.sidebar:
    with st.form("search_form"):
        city_input = st.text_input("Enter a city name")
        submitted = st.form_submit_button("Search")

if submitted:
    results = search_city(city_input)
    if results is None:
        st.sidebar.warning(f"No results found for '{city_input}'.")
        st.session_state.results = None
    else:
        st.session_state.results = results
        st.session_state.selected_city = None

# --- Stage 2: select correct city ---
if st.session_state.results:
    with st.sidebar:
        select_correct_city(st.session_state.results)
    #city = st.session_state.results
    #st.write('You selected ', city)

# --- Stage 3: get city details, incl. coordinates ---
city_details = get_city_details()


if city_details:
    if city_details['admin1'] == '':
        city = city_details['name'] + ', ' + city_details['country']
    else:
        city = city_details['name'] + ', ' + city_details['admin1'] + ', ' + city_details['country']
    latitude = city_details['latitude']
    longitude = city_details['longitude']
    st.header(city)
#else:
#    raise ValueError('No coordinates found.')

# --- Stage 4: Select and create plot ---
    plot_options = [
                    'Daily maximum temperatures',
                    'Temperatures during heatwaves (5°C)',
                    'Temperatures during heatwaves (std)',
                    'Number of heatwave days per year (5°C)',
                    'Number of heatwave days per year (std)',
                    'Copernicus spaghetti plot',
                    'Warming stripes'
                    ]
    selected = st.sidebar.selectbox(label='Which plot are you interested in?', options=plot_options, key="my_late_selectbox")
    st.markdown("""
        <style>
        div[data-testid="stSelectbox"]:has(input[aria-labelledby*="my_late_selectbox"]) div[data-baseweb="popover"] {
            transform: translateY(-100%) !important;
        }
        </style>
        """, unsafe_allow_html=True)

    if selected == 'Daily maximum temperatures':
        daily_max_plot(latitude, longitude, select_years())

    elif selected == 'Temperatures during heatwaves (5°C)':
       heatwave_days_plot(latitude, longitude, 'fixed', select_years())
    
    elif selected == 'Temperatures during heatwaves (std)':
        heatwave_days_plot(latitude, longitude, 'std',  select_years())

    elif selected == 'Number of heatwave days per year (5°C)':
        heatwave_count_plot(latitude, longitude, 'fixed')

    elif selected == 'Number of heatwave days per year (std)':
        heatwave_count_plot(latitude, longitude, 'std')

    elif selected == 'Copernicus spaghetti plot':
        spaghetti_plot(latitude, longitude, select_years())

    elif selected == 'Warming stripes':
        warming_stripes_plot(latitude, longitude)



