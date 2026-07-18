import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st # for web applet

# Change font size across the applet
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 18px;
    }
    label, .stSelectbox label, [data-testid="stWidgetLabel"] {
        font-size: 22px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Change font size for plots
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rcParams.update({'font.size': BIGGER_SIZE})

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#---------------------------------------------------
# LOADING AND PREPARING THE DATA

@st.cache_data
def load_data():
    # CO2 fossil emissions
    # https://edgar.jrc.ec.europa.eu/dataset_ghg2025
    co2 = pd.read_excel(
        'Kaya-decomposition/IEA_EDGAR_CO2_1970_2024.xlsx',
        sheet_name = 'TOTALS BY COUNTRY',
        skiprows = 9
        )
        # In this dataset, I had to manually rename 'Czech Republic' to 'Czechia'

    # Population
    # https://data.worldbank.org/indicator/SP.POP.TOTL
    population = pd.read_csv(
        'Kaya-decomposition/population.csv',
        skiprows = 4
    )

    # GDP per capita annual growth in %
    # https://data.worldbank.org/indicator/NY.GDP.PCAP.KD.ZG
    gdp_population = pd.read_csv(
        'Kaya-decomposition/GDP_PCAP.csv',
        skiprows = 4
    )

    # Energy use (kg of oil equivalent) per $1,000 GDP (constant 2021 PPP)
    # https://data.worldbank.org/indicator/EG.USE.COMM.GD.PP.KD
    energy_gdp = pd.read_csv(
        'Kaya-decomposition/Energy_perGDP.csv',
        skiprows = 4
    )

    # Carbon intensity of energy
    # https://ourworldindata.org/grapher/co2-per-unit-energy

    co2_energy_long = pd.read_csv(
        'Kaya-decomposition/co2-per-unit-energy.csv'
    )

    #---------------------------------------------------
    # Rename and delete columns, melt dataframes 

    co2 = co2.drop(columns=['IPCC_annex', 'C_group_IM24_sh', 'Substance'])
    co2 = co2.drop(co2.columns[2:22], axis=1)
    co2.columns = [col.replace('Y_', '') for col in co2.columns] 
        # Delete 'Y_' in the name of the year to match other datasets
    co2 = co2.rename(columns={'Name' : 'Country', 'Country_code_A3' : 'Country Code'})
    co2_long = co2.melt(id_vars=['Country', 'Country Code'], var_name='year', value_name='value')
    co2_long['unit'] = 'Fossil CO2'

    population = population.drop(columns=['Indicator Name', 'Indicator Code', '2025'])
    population = population.drop(population.columns[2:32], axis=1)
    population = population.rename(columns={'Country Name' : 'Country'})
    population_long = population.melt(id_vars=['Country', 'Country Code'], var_name='year', value_name='value')
    population_long['unit'] = 'Population'

    gdp_population = gdp_population.drop(columns=['Indicator Name', 'Indicator Code',  'Unnamed: 70', '2025'])
    gdp_population = gdp_population.drop(gdp_population.columns[2:32], axis=1)
    gdp_population = gdp_population.rename(columns={'Country Name' : 'Country'})
    gdp_population_long = gdp_population.melt(id_vars=['Country', 'Country Code'], var_name='year', value_name='value')
    gdp_population_long['unit'] = 'GDP/Population'

    energy_gdp = energy_gdp.drop(columns=['Indicator Name', 'Indicator Code', '2025'])
    energy_gdp = energy_gdp.drop(energy_gdp.columns[2:32], axis=1)
    energy_gdp = energy_gdp.rename(columns={'Country Name' : 'Country'})
    energy_gdp_long = energy_gdp.melt(id_vars=['Country', 'Country Code'], var_name='year', value_name='value')
    energy_gdp_long['unit'] = 'Energy/GDP'

    co2_energy_long['unit'] = 'CO2/Energy'
    co2_energy_long = co2_energy_long.rename(columns={'Entity' : 'Country', 'Code' : 'Country Code', 'CO₂ emissions per unit energy' : 'value', 'Year' : 'year'})
    co2_energy_long = co2_energy_long.drop(
        co2_energy_long[co2_energy_long['year'] < 1991].index
        )
        # delete rows with data for years before 1991
    co2_energy_long = co2_energy_long.drop(
                        co2_energy_long[co2_energy_long['Country Code'].isna()].index
                    )
    co2_energy_long = co2_energy_long.drop(
        co2_energy_long[co2_energy_long['Country'].isin([
            'Africa', 
            'Asia', 
            'Asia (excl. China and India)', 
            'Oceania', 
            'North America', 
            'Europe', 
            'Europe (excl. EU-27)', 
            'Europe (excl. EU-28)', 
            'European Union (27)', 
            'European Union (28)', 
            'South America', 
            'Antarctica', 
            'High-income countries', 
            'Low-income countries', 
            'Lower-middle-income countries', 
            'Upper-middle-income countries',  
            'World',
            'Kosovo'
            ])].index
        )
    co2_energy_long = co2_energy_long.astype({'year' : 'str'}) 
        # All other tables have years as strings

    #---------------------------------------------------
    # Concat the data

    all_countries = pd.concat([
                                co2_long, 
                                population_long, 
                                gdp_population_long, 
                                energy_gdp_long, 
                                co2_energy_long
                            ])

    return all_countries, co2_energy_long

#---------------------------------------------------
# PREPARE COUNTRY SPECIFIC DATA

def country_data(df, country):
    # Filters country data 
    df_country = df[df['Country Code'] == country_to_code[country]]
    df_country_pivot = df_country.pivot(index='year', columns='unit', values='value')
    df_country_pivot = df_country_pivot.reindex(columns=['Population', 'GDP/Population', 'Energy/GDP', 'CO2/Energy', 'Fossil CO2'])
    df_country_pivot.index = df_country_pivot.index.astype(int)
    return df_country_pivot

def country_data_for_kaya(df, country):
    # Filters country data and prepares them for plotting kaya decomposition
    df_country_pivot = country_data(df, country)

    # Take % growth
    # Careful, GDP/Population is already in %
    df_country_pivot.iloc[:,0:1] = df_country_pivot.iloc[:,0:1].pct_change(axis=0, fill_method=None) * 100
    df_country_pivot.iloc[:,2:] = df_country_pivot.iloc[:,2:].pct_change(axis=0, fill_method=None) * 100
    df_country_pivot = df_country_pivot.iloc[1:,:] # Deletes first row that is filled with NaN
    return df_country_pivot

#---------------------------------------------------
# PLOT FUNCTION

def kayaplot(all_data, country):
    df = country_data_for_kaya(all_data, country)

    # df index = years, columns = ['Population', 'GDP/Population', 'Energy/GDP', 'CO2/Energy', 'Fossil CO2']
    # Values are % growth rates (e.g. 2.1 for 2.1%)

    fig, ax = plt.subplots(figsize=(12, 6))

    factors = ['Population', 'GDP/Population', 'Energy/GDP', 'CO2/Energy']
    colors = ['#e07b54', '#f0c05a', "#84c886", "#9d6bd6"]

    bottoms_pos = pd.Series(0.0, index=df.index)
    bottoms_neg = pd.Series(0.0, index=df.index)

    for factor, color in zip(factors, colors):
        values = df[factor]
        pos = values.clip(lower=0)
        neg = values.clip(upper=0)
        ax.bar(df.index, pos, bottom=bottoms_pos, color=color, label=factor)
        ax.bar(df.index, neg, bottom=bottoms_neg, color=color)
        bottoms_pos += pos
        bottoms_neg += neg

    # Overlay actual CO2 change
    ax.plot(df.index, df['Fossil CO2'], color='black', marker='o', linestyle='none', label='Fossil CO2')

    # Formatting
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(df.index[::5])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=1))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title('Kaya Decomposition for {}'.format(country))
    ax.legend()

    st.pyplot(fig)

#---------------------------------------------------
# RUN THE CODE
all_countries, co2_energy_long = load_data()

# Create a list of countries (based on the shortest dataset)
countries = co2_energy_long['Country'].unique().tolist()

# Create dictionary {country : code}
cc = co2_energy_long[['Country', 'Country Code']].drop_duplicates().to_records(index=False)
country_to_code = {country: code for (country, code) in cc}

#st.markdown("<p style='font-size:22px;'>Select a country</p>", unsafe_allow_html=True)
selected = st.selectbox(label='Select a country', options=countries, index=countries.index('Italy'))
kayaplot(all_countries, selected)
