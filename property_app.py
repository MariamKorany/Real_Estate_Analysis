
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your property data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('property_data_08152025_v2.csv')
        return df
    except FileNotFoundError:
        st.error(" Data file not found! Please make sure 'property_data_08152025_v2.csv' is in the same folder.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

# Only proceed if data is loaded successfully
if df is not None:
    # Sidebar navigation
    st.sidebar.title(" Property Analysis App")
    page_option = st.sidebar.radio(
        label='Navigate to:', 
        options=['Data Explorer', 'Interactive Plots']
    )

    def data_page():
        st.title("Property Data Explorer")
        st.write("### Overview of Egyptian Property Market Data")

        # Show basic info with error handling
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Properties", f"{len(df):,}")

        with col2:
            if 'property_type' in df.columns:
                st.metric("Property Types", df['property_type'].nunique())
            else:
                st.metric("Property Types", "N/A")

        with col3:
            if 'price' in df.columns:
                avg_price = df['price'].mean()
                if not np.isnan(avg_price):
                    st.metric("Average Price", f"{avg_price:,.0f} EGP")
                else:
                    st.metric("Average Price", "N/A")
            else:
                st.metric("Average Price", "N/A")

        with col4:
            if 'property_size' in df.columns:
                avg_size = df['property_size'].mean()
                if not np.isnan(avg_size):
                    st.metric("Average Size", f"{avg_size:.0f} sqm")
                else:
                    st.metric("Average Size", "N/A")
            else:
                st.metric("Average Size", "N/A")

        # Show data overview
        st.write("### Dataset Overview")
        st.write(f"**Rows:** {df.shape[0]:,} | **Columns:** {df.shape[1]}")

        # Show column info
        st.write("### Available Columns")
        st.write(", ".join(df.columns.tolist()))

        # Show sample data
        st.write("### Sample Data")
        st.dataframe(df.head())

        # Property type distribution
        if 'property_type' in df.columns:
            st.write("### Property Type Distribution")
            prop_counts = df['property_type'].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(prop_counts.head(10))
            with col2:
                fig = px.pie(values=prop_counts.values, names=prop_counts.index, 
                           title="Property Types Distribution")
                st.plotly_chart(fig, use_container_width=True)

        # Filter options
        if 'property_type' in df.columns:
            st.write("### Filter and Explore")
            property_types = st.multiselect(
                "Filter by Property Type:",
                options=df['property_type'].unique(),
                default=df['property_type'].unique()[:min(3, len(df['property_type'].unique()))]
            )

            if property_types:  # Only filter if something is selected
                # Apply filters
                filtered_df = df[df['property_type'].isin(property_types)]

                # Show filtered data
                display_columns = []
                for col in ['property_type', 'location', 'bedrooms', 'bathrooms', 'price', 'property_size', 'finishing']:
                    if col in df.columns:
                        display_columns.append(col)

                if display_columns:
                    st.dataframe(filtered_df[display_columns])

                # Summary statistics for numeric columns
                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.write("### Summary Statistics")
                    st.dataframe(filtered_df[numeric_cols].describe())

    def plots_page():
        st.title("Interactive Property Analysis")

        # Check available columns
        available_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Plot selection
        plot_type = st.selectbox(
            "Choose Analysis Type:",
            ['Price Distribution', 'Property Comparison', 'Location Analysis']
        )

        if plot_type == 'Price Distribution':
            st.subheader("Price Analysis")

            if 'price' not in df.columns:
                st.error("Price column not found in the dataset!")
                return

            # X-axis selection from categorical columns
            cat_options = [col for col in ['property_type', 'finishing', 'sale_type', 'compound'] 
                          if col in categorical_cols]

            if cat_options:
                x_option = st.selectbox(
                    'Select Category for Analysis:', 
                    options=cat_options, 
                    index=0
                )

                if st.button("Show Price Distribution"):
                    # Create box plot
                    fig = px.box(df, x=x_option, y='price', 
                                title=f'Price Distribution by {x_option.replace("_", " ").title()}')
                    st.plotly_chart(fig, use_container_width=True)

                    # Show summary stats
                    st.write(f"### Price Summary by {x_option.replace('_', ' ').title()}")
                    summary = df.groupby(x_option)['price'].agg(['mean', 'median', 'count']).round(2)
                    st.dataframe(summary)
            else:
                st.warning("No suitable categorical columns found for price analysis.")

        elif plot_type == 'Property Comparison':
            st.subheader("Property Features Analysis")

            if 'price' not in df.columns:
                st.error("Price column not found in the dataset!")
                return

            # Multi-variable analysis
            numeric_options = [col for col in ['bedrooms', 'bathrooms', 'property_size'] 
                              if col in numeric_cols]
            color_options = [col for col in ['property_type', 'finishing', 'sale_type'] 
                            if col in categorical_cols]

            if numeric_options and color_options:
                x_var = st.selectbox('X-axis:', numeric_options)
                color_var = st.selectbox('Color by:', color_options)

                if st.button("Show Comparison"):
                    hover_cols = [col for col in ['location', 'compound'] if col in df.columns]
                    fig = px.scatter(df, x=x_var, y='price', color=color_var,
                                   title=f'Price vs {x_var.title()} (Colored by {color_var.replace("_", " ").title()})',
                                   hover_data=hover_cols)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient columns for property comparison.")

        elif plot_type == 'Location Analysis':
            st.subheader("Location-Based Analysis")

            if 'location' not in df.columns:
                st.error("Location column not found in the dataset!")
                return

            # Extract city from location
            df_temp = df.copy()
            df_temp['city'] = df_temp['location'].str.split(',').str[1].str.strip()

            # Top cities by property count
            top_cities = df_temp['city'].value_counts().head(10)

            col1, col2 = st.columns(2)

            with col1:
                st.write("### Top Cities by Property Count")
                fig1 = px.bar(x=top_cities.index, y=top_cities.values,
                             title='Properties per City')
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                if 'price' in df.columns:
                    st.write("### Average Price by City")
                    city_prices = df_temp.groupby('city')['price'].mean().sort_values(ascending=False).head(10)
                    fig2 = px.bar(x=city_prices.index, y=city_prices.values,
                                 title='Average Price per City')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.write("Price column not available for city analysis.")

    # Page routing
    if page_option == 'Data Explorer':
        data_page()
    elif page_option == 'Interactive Plots':
        plots_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with using Streamlit")
    st.sidebar.markdown("Property Data Analysis Dashboard")

else:
    st.title("Data Loading Error")
    st.write("Please check:")
    st.write("1. Make sure your CSV file exists")
    st.write("2. Check the file path and name")
    st.write("3. Ensure the file is not corrupted")
