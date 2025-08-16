import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure Streamlit
st.set_page_config(page_title="Real Estate EDA", layout="wide")
sns.set(style="whitegrid")

# ----------------- Load dataset -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("property_data_08152025_v2.csv")

    # Standardize column name
    if "price" in df.columns and "price (EGP)" not in df.columns:
        df.rename(columns={"price": "price (EGP)"}, inplace=True)

    # Clean price column once (numeric conversion)
    df["price (EGP)"] = (
        df["price (EGP)"]
        .astype(str)
        .str.replace(",", "", regex=True)
        .str.replace("EGP", "", regex=True)
        .str.replace("[^0-9.]", "", regex=True)
        .str.strip()
    )
    df["price (EGP)"] = pd.to_numeric(df["price (EGP)"], errors="coerce")

    # Clean property_size too (for Q8)
    if "property_size" in df.columns:
        df["property_size"] = pd.to_numeric(df["property_size"], errors="coerce")

    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Description", "EDA & Visuals"])

# ----------------- Page 1 -----------------
if page == "Description":
    st.title("🏠 Real Estate Market Analysis")
    st.write("""
    This app provides an exploratory data analysis (EDA) of the Egyptian real estate market.  

    **EDA Questions:**
    1. Distribution of property prices  
    2. Most common property type  
    3. Properties by number of bedrooms  
    4. Developers with the most listings  
    5. Areas with most properties listed  
    6. Sale vs Rent distribution  
    7. Average price by property type  
    8. Relationship between size & price  
    9. Price distribution by delivery status  
    10. Most expensive developers  
    11. Impact of finishing & property type on price  
    12. Top locations by average price & property type  
    13. Location–sale type combinations commanding highest prices  
    """)

# ----------------- Page 2 -----------------
elif page == "EDA & Visuals":
    st.title("🔎 Exploratory Data Analysis")

    # Dropdown to choose question
    question = st.selectbox(
        "Choose an analysis question:",
        [
            "Q1: Distribution of Property Prices",
            "Q2: Most Common Property Type",
            "Q3: Distribution of Bedrooms",
            "Q4: Developers with Most Properties",
            "Q5: Areas with Most Properties",
            "Q6: Sale Type Distribution",
            "Q7: Average Price by Property Type",
            "Q8: Property Size vs Price",
            "Q9: Price Distribution by Delivery Status",
            "Q10: Most Expensive Developers",
            "Q11: Price by Finishing & Property Type",
            "Q12: Top Locations by Avg Price & Property Type",
            "Q13: Prices by Location & Sale Type"
        ]
    )

    # Q1: Distribution of Property Prices
    if question == "Q1: Distribution of Property Prices":
        st.subheader("Distribution of Property Prices")

        # Clean data: remove NaN and zero values
        clean_prices = df['price (EGP)'].dropna()
        clean_prices = clean_prices[clean_prices > 0]

        # Display min and max prices
        st.write(f"**Minimum price:** {clean_prices.min() / 1e6:.1f} Million EGP")
        st.write(f"**Maximum price:** {clean_prices.max() / 1e6:.1f} Million EGP")

        fig, ax = plt.subplots(figsize=(13,6))
        sns.histplot(clean_prices / 1e6, bins=50, kde=True, ax=ax)
        ax.set_title('Distribution of Property Prices')
        ax.set_xlabel('Price (Million EGP)')
        ax.set_ylabel('Frequency')

        # Format x-axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        plt.xticks(rotation=45)
        plt.locator_params(axis='x', nbins=15)  # Adjusted for streamlit
        plt.tight_layout()
        st.pyplot(fig)

        st.info("📊 This histogram shows the distribution of property prices in millions. "
                "It helps identify common price ranges, skewness, and potential outliers.")

    # Q2: Most Common Property Type
    elif question == "Q2: Most Common Property Type":
        st.subheader("Most Common Property Type")
        fig, ax = plt.subplots(figsize=(8,5))
        df['property_type'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Property Type Distribution')
        ax.set_xlabel('Property Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # Q3: Distribution of Bedrooms
    elif question == "Q3: Distribution of Bedrooms":
        st.subheader("Distribution of Bedrooms")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(x='bedrooms', data=df, order=sorted(df['bedrooms'].dropna().unique()), ax=ax)
        ax.set_title("Distribution of Bedrooms")
        ax.set_xlabel("Bedrooms")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Q4: Developers with Most Properties
    elif question == "Q4: Developers with Most Properties":
        st.subheader("Developers with Most Properties")

        # Set style
        sns.set(style="whitegrid", palette="pastel")

        # Top 10 developers by number of properties
        top_dev = df['developer_name'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(12,7))
        sns.barplot(x=top_dev.values, y=top_dev.index, ax=ax)
        ax.set_title("Top 10 Developers by Number of Properties for Sale", fontsize=14, pad=20)
        ax.set_xlabel("Number of Properties Available", fontsize=12)
        ax.set_ylabel("Developer Name", fontsize=12)

        # Add more x-axis ticks
        plt.locator_params(axis='x', nbins=10)

        # Add value labels on bars
        for i, (developer, count) in enumerate(top_dev.items()):
            ax.text(count + max(top_dev.values) * 0.01, i, str(count), 
                    va='center', ha='left', fontweight='bold', fontsize=10)

        # Add grid
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Print summary statistics
        st.write("**Summary:**")
        st.write(f"- Total unique developers: {df['developer_name'].nunique()}")
        st.write(f"- Total properties: {len(df):,}")
        st.write(f"- Average properties per developer: {len(df) / df['developer_name'].nunique():.1f}")
        st.write(f"- Top developer ({top_dev.index[0]}) has {top_dev.iloc[0]} properties")
        st.write(f"- This represents {(top_dev.iloc[0] / len(df) * 100):.1f}% of all properties")

    # Q5: Areas with Most Properties
    elif question == "Q5: Areas with Most Properties":
        st.subheader("Areas with Most Properties")

        # Top 10 locations
        top_loc = df['location'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=top_loc.values, y=top_loc.index, ax=ax)
        ax.set_title("Which areas have the most properties listed?")
        ax.set_xlabel("Number of Listings")
        ax.set_ylabel("Location")
        st.pyplot(fig)

    # Q6: Sale Type Distribution
    elif question == "Q6: Sale Type Distribution":
        st.subheader("Property Sale Types Distribution")

        sale_counts = df['sale_type'].value_counts()

        # Pie chart with better visibility
        fig, ax = plt.subplots(figsize=(10,8))

        # Create pie chart with exploded slices
        explode = [0.05] * len(sale_counts)
        colors = plt.cm.Set3(range(len(sale_counts)))

        wedges, texts, autotexts = ax.pie(sale_counts, 
                                         labels=sale_counts.index, 
                                         autopct='%1.1f%%', 
                                         startangle=90, 
                                         colors=colors,
                                         explode=explode,
                                         textprops={'fontsize': 12, 'fontweight': 'bold'},
                                         pctdistance=0.85)

        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)

        # Add count values to the legend
        legend_labels = [f'{label}: {count:,} properties' for label, count in sale_counts.items()]
        ax.legend(wedges, legend_labels, title="Sale Types", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)

        ax.set_title('Distribution of Sale vs Rent Properties', fontsize=14, pad=20)
        plt.tight_layout()
        st.pyplot(fig)

        # Print detailed breakdown
        st.write("**Detailed Breakdown:**")
        total = sale_counts.sum()
        for sale_type, count in sale_counts.items():
            percentage = (count / total) * 100
            st.write(f"- {sale_type}: {count:,} properties ({percentage:.1f}%)")

    # Q7: Average Price by Property Type
    elif question == "Q7: Average Price by Property Type":
        st.subheader("Average Price by Property Type")

        avg_price_type = df.groupby('property_type')['price (EGP)'].mean().sort_values()

        fig, ax = plt.subplots(figsize=(8,5))
        avg_price_type.plot(kind='barh', ax=ax)
        ax.set_title("Average Price by Property Type")
        ax.set_xlabel("Average Price (Million EGP)")
        ax.set_ylabel("Property Type")

        # Format x-axis in millions
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        plt.locator_params(axis='x', nbins=7)
        st.pyplot(fig)

    # Q8: Property Size vs Price
    elif question == "Q8: Property Size vs Price":
        st.subheader("Property Size vs Price")

        # Ensure numeric values
        df['property_size'] = pd.to_numeric(df['property_size'], errors='coerce')
        df['price (EGP)'] = pd.to_numeric(df['price (EGP)'], errors='coerce')

        # Drop rows where either value is NaN
        df_clean = df.dropna(subset=['property_size', 'price (EGP)'])

        fig, ax = plt.subplots(figsize=(12,7))
        sns.scatterplot(x='property_size', y='price (EGP)', data=df_clean, alpha=0.6, ax=ax)
        ax.set_title("Property Size vs Price")
        ax.set_xlabel("Size (m²)")
        ax.set_ylabel("Price (Million EGP)")

        # Format y-axis in millions
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        plt.locator_params(axis='y', nbins=12)
        plt.locator_params(axis='x', nbins=10)
        st.pyplot(fig)

# Q9: Price Distribution by Delivery Status (IMPROVED - CLEARER)
    elif question == "Q9: Price Distribution by Delivery Status":
        st.subheader("Price Distribution by Delivery Status")

        # Clean and categorize delivery status
        df_delivery = df.dropna(subset=['delivery_in', 'price (EGP)'])

        # Categorize delivery status for better understanding
        def categorize_delivery(delivery):
            delivery_str = str(delivery).lower()
            if 'delivered' in delivery_str or 'ready' in delivery_str:
                return 'Ready/Delivered'
            elif delivery_str.isdigit():
                year = int(delivery_str)
                if year <= 2025:
                    return '2025 or earlier'
                elif year <= 2027:
                    return '2026-2027'
                else:
                    return '2028 and later'
            else:
                return 'Other/Unspecified'

        df_delivery['delivery_category'] = df_delivery['delivery_in'].apply(categorize_delivery)

        # Create violin plot and box plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))

        # Left: Violin plot for distribution shape
        sns.violinplot(data=df_delivery, x='delivery_category', y='price (EGP)', ax=ax1)
        ax1.set_title("Price Distribution by Delivery Category (Violin Plot)")
        ax1.set_xlabel("Delivery Category")
        ax1.set_ylabel("Price (Million EGP)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        ax1.tick_params(axis='x', rotation=45)

        # Right: Average prices by category
        avg_prices = df_delivery.groupby('delivery_category')['price (EGP)'].agg(['mean', 'count']).reset_index()
        sns.barplot(data=avg_prices, x='mean', y='delivery_category', ax=ax2)
        ax2.set_title("Average Price by Delivery Category")
        ax2.set_xlabel("Average Price (Million EGP)")
        ax2.set_ylabel("Delivery Category")
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

        # Add count labels on bars
        for i, row in avg_prices.iterrows():
            ax2.text(row['mean'] + avg_prices['mean'].max() * 0.02, i, 
                    f"({int(row['count'])} properties)", va='center', ha='left', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        # Show detailed statistics
        st.write("**Price Statistics by Delivery Category:**")
        for category in avg_prices['delivery_category']:
            cat_data = df_delivery[df_delivery['delivery_category'] == category]['price (EGP)']
            st.write(f"- **{category}**: Avg: {cat_data.mean()/1e6:.2f}M EGP, "
                    f"Median: {cat_data.median()/1e6:.2f}M EGP, "
                    f"Count: {len(cat_data)} properties")

        highest_avg_category = avg_prices.loc[avg_prices['mean'].idxmax(), 'delivery_category']
        st.info(f"📊 **Insights:** {highest_avg_category} properties command the highest average prices, "
                "indicating the premium buyers pay for immediate availability or the price appreciation "
                "over time. Pre-construction properties may offer better value for money but carry delivery risk.")

    # Q10: Most Expensive Developers
    elif question == "Q10: Most Expensive Developers":
        st.subheader("Developers with Highest Average Prices")

        # Top 10 developers by average price
        top_dev_prices = (df.groupby('developer_name')['price (EGP)']
                            .mean()
                            .sort_values(ascending=False)
                            .head(10))

        fig, ax = plt.subplots(figsize=(12,7))
        sns.barplot(x=top_dev_prices.values, y=top_dev_prices.index, palette='viridis', ax=ax)
        ax.set_xlabel('Average Price (Million EGP)', fontsize=12)
        ax.set_ylabel('Developer Name', fontsize=12)
        ax.set_title('Top 10 Developers by Average Property Price', fontsize=14, pad=20)

        # Format x-axis in millions
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        plt.locator_params(axis='x', nbins=10)

        # Add value labels on bars
        for i, (developer, price) in enumerate(top_dev_prices.items()):
            ax.text(price + max(top_dev_prices.values) * 0.01, i, f'{price/1e6:.1f}M',
                    va='center', ha='left', fontweight='bold', fontsize=10)

        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Print summary
        st.write("**Summary:**")
        st.write(f"- Most expensive developer: {top_dev_prices.index[0]} (Avg: {top_dev_prices.iloc[0]/1e6:.1f}M EGP)")
        st.write(f"- Least expensive in top 10: {top_dev_prices.index[-1]} (Avg: {top_dev_prices.iloc[-1]/1e6:.1f}M EGP)")
        st.write(f"- Price range difference: {(top_dev_prices.iloc[0] - top_dev_prices.iloc[-1])/1e6:.1f}M EGP")
        st.write(f"- Overall average price: {df['price (EGP)'].mean()/1e6:.1f}M EGP")

 # Q11: Price by Finishing & Property Type (IMPROVED - CLEARER)
    elif question == "Q11: Price by Finishing & Property Type":
        st.subheader("Price by Finishing & Property Type")

        # Clean data and create a more focused analysis
        df_finishing = df.dropna(subset=['finishing', 'property_type', 'price (EGP)'])

        # Create a pivot table for average prices
        price_pivot = df_finishing.pivot_table(values='price (EGP)', 
                                             index='finishing', 
                                             columns='property_type', 
                                             aggfunc='mean')

        # Create subplots for different views
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,12))

        # Top left: Heatmap of average prices
        sns.heatmap(price_pivot / 1e6, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Avg Price (M EGP)'}, ax=ax1)
        ax1.set_title('Average Price by Finishing & Property Type')
        ax1.set_xlabel('Property Type')
        ax1.set_ylabel('Finishing Level')

        # Top right: Average price by finishing (overall)
        finishing_avg = df_finishing.groupby('finishing')['price (EGP)'].mean().sort_values()
        sns.barplot(x=finishing_avg.values, y=finishing_avg.index, ax=ax2)
        ax2.set_title('Average Price by Finishing Level')
        ax2.set_xlabel('Average Price (Million EGP)')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

        # Bottom left: Count of properties by finishing and type
        finishing_counts = df_finishing.groupby(['finishing', 'property_type']).size().reset_index(name='count')
        sns.barplot(data=finishing_counts, x='finishing', y='count', hue='property_type', ax=ax3)
        ax3.set_title('Number of Properties by Finishing & Type')
        ax3.set_xlabel('Finishing Level')
        ax3.set_ylabel('Count')
        ax3.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Bottom right: Box plot for price distribution
        sns.boxplot(data=df_finishing, x='finishing', y='price (EGP)', ax=ax4)
        ax4.set_title('Price Distribution by Finishing Level')
        ax4.set_xlabel('Finishing Level')
        ax4.set_ylabel('Price (Million EGP)')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

        plt.tight_layout()
        st.pyplot(fig)

        # Show insights based on the analysis
        highest_finishing = finishing_avg.index[-1]
        lowest_finishing = finishing_avg.index[0]
        price_difference = finishing_avg.iloc[-1] - finishing_avg.iloc[0]

        st.write("**Key Statistics:**")
        st.write(f"- Highest average price finishing: {highest_finishing} ({finishing_avg.iloc[-1]/1e6:.1f}M EGP)")
        st.write(f"- Lowest average price finishing: {lowest_finishing} ({finishing_avg.iloc[0]/1e6:.1f}M EGP)")
        st.write(f"- Price premium for best finishing: {price_difference/1e6:.1f}M EGP ({(price_difference/finishing_avg.iloc[0]*100):.1f}% increase)")

        st.info(f"📊 **Insights:** {highest_finishing} finishing commands a premium of "
                f"{price_difference/1e6:.1f}M EGP over {lowest_finishing} finishing, "
                f"representing a {(price_difference/finishing_avg.iloc[0]*100):.1f}% price increase. "
                "The heatmap reveals that finishing level significantly impacts pricing across all property types, "
                "with luxury finishes being a key value driver in the Egyptian real estate market.")

    # Q12: Top Locations by Avg Price & Property Type
    elif question == "Q12: Top Locations by Avg Price & Property Type":
        st.subheader("Top Locations by Avg Price & Property Type")

        top_locations = df.groupby(['location','property_type'])['price (EGP)'].mean().reset_index()
        top_locations = top_locations.sort_values(by='price (EGP)', ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x='price (EGP)', y='location', hue='property_type', data=top_locations, ax=ax)
        ax.set_title("Top Locations by Average Price & Property Type", fontsize=12, pad=15)
        ax.set_xlabel("Average Price (Million EGP)", fontsize=10)
        ax.set_ylabel("Location", fontsize=10)

        # Format x-axis in millions
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        plt.locator_params(axis='x', nbins=6)

        # Compact legend positioning
        ax.legend(title="Property Type", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        ax.grid(axis='x', alpha=0.2)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        st.pyplot(fig)

    # Q13: Prices by Location & Sale Type
    elif question == "Q13: Prices by Location & Sale Type":
        st.subheader("Prices by Location & Sale Type")

        # Create pivot table
        price_heatmap_data = df.pivot_table(values='price (EGP)', 
                                           index='location', 
                                           columns='sale_type', 
                                           aggfunc='mean')

        # Get top 10 locations by average price
        top_locations = df.groupby('location')['price (EGP)'].mean().sort_values(ascending=False).head(10)
        price_heatmap_filtered = price_heatmap_data.loc[top_locations.index]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(price_heatmap_filtered / 1e6,  
                    annot=True,  
                    fmt='.1f',   
                    cmap='YlOrRd',  
                    cbar_kws={'label': 'Avg Price (M EGP)'},  
                    linewidths=0.5,
                    square=False,  
                    annot_kws={'size': 9}, ax=ax)  

        ax.set_title('Property Prices by Location & Sale Type (Top 10)', fontsize=12, pad=15)
        ax.set_xlabel('Sale Type', fontsize=10)
        ax.set_ylabel('Location', fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        # Find and display highest price combinations
        st.write("**Top 10 Location-Sale Type Combinations by Average Price:**")

        # Flatten the pivot table to find top combinations
        location_sale_prices = []
        for location in price_heatmap_data.index:
            for sale_type in price_heatmap_data.columns:
                price = price_heatmap_data.loc[location, sale_type]
                if not pd.isna(price):
                    location_sale_prices.append({
                        'location': location,
                        'sale_type': sale_type,
                        'avg_price': price
                    })

        # Sort and display top combinations
        top_combinations = sorted(location_sale_prices, key=lambda x: x['avg_price'], reverse=True)[:10]

        for i, combo in enumerate(top_combinations, 1):
            st.write(f"{i:2d}. {combo['location']} - {combo['sale_type']}: "
                    f"{combo['avg_price']/1e6:.1f}M EGP")

        # Calculate insights
        rental_avg = df[df['sale_type'].str.contains('rent', case=False, na=False)]['price (EGP)'].mean() if any(df['sale_type'].str.contains('rent', case=False, na=False)) else 0
        sale_avg = df[df['sale_type'].str.contains('sale', case=False, na=False)]['price (EGP)'].mean() if any(df['sale_type'].str.contains('sale', case=False, na=False)) else 0

        st.write("**Market Insights:**")
        st.write(f"- Average rental price: {rental_avg/1e6:.1f}M EGP")
        st.write(f"- Average sale price: {sale_avg/1e6:.1f}M EGP")

        st.info("📊 This heatmap compares average property prices across locations "
                "and sale types. Darker shades indicate higher prices.")
