import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class SteamLit:
    def __init__(self):
        self.ISIN = None
        self.date_range1 = None
        self.date_range2 = None
        self.max_length = 200  # Modify as needed
        
        if 'run' not in st.session_state:
            st.session_state['run'] = False    
        
    def click_button(self):
        st.session_state['run'] = True
        
    def display_input(self):
        # self.ISIN = st.selectbox('ISIN',tuple(self.fund_nav_df["ISIN"].unique().tolist()))
        self.ISIN = st.selectbox('ISIN',('61500003','61500003'))
        self.date_range1 = st.slider("Date Range 1", min_value=1, max_value=self.max_length, value=30, step=1)
        self.date_range2 = st.slider("Date Range 2", min_value=1, max_value=self.max_length//self.date_range1, value=20, step=1)
        st.button("Run", on_click=self.click_button)
        # self.option = st.multiselect(
        # "Select three known variables:",
        # self.factor_df.columns.tolist().remove('date'),
        # )
    def get_data(self):
        self.fund_nav_df = pd.read_csv("Data_Fund_NAV_new.csv")
        self.fund_nav_df["ISIN"] = self.fund_nav_df["ISIN"].astype(str)
        self.factor_df = pd.read_csv("allfactors.csv")
        # self.factor_df = self.factor_df[self.option]
        
    def pre_preprocess(self):
        self.factor_df['date'] = pd.to_datetime(self.factor_df['date'])
        self.fund_nav_df = self.fund_nav_df[self.fund_nav_df["ISIN"] == self.ISIN]
        self.fund_nav_df['date'] = pd.to_datetime(self.fund_nav_df['date'])
        self.fund_nav_df["log_returns"] = np.log(self.fund_nav_df.value / self.fund_nav_df.value.shift(1))
        self.df_beta = pd.merge(self.fund_nav_df[['date', 'log_returns']], self.factor_df, on='date', how='left').dropna()



        
    def fit(self):
        list_factor_date = []    
        
        for n in range(0, len(self.df_beta), self.date_range1):

            df_beta_ = self.df_beta.iloc[n:n+self.date_range2]
            y = df_beta_.iloc[:, 1:2].to_numpy()
            X = df_beta_.iloc[:, 2:-1]

            reg = LinearRegression(positive=True).fit(X, y)
            liber_coeff = pd.DataFrame(reg.coef_, columns=reg.feature_names_in_)
            R2 = r2_score(y, reg.predict(X))
            liber_coeff["R2"] = R2
            liber_coeff["Alpha"] = reg.intercept_
            liber_coeff["date"] = df_beta_.date.iloc[-1]
            list_factor_date.append(liber_coeff)
            # print(liber_coeff)
        self.df_factor_time = pd.concat(list_factor_date)
        self.df_factor_time.set_index('date', inplace=True)    
        
    def plot(self):

        st.write('Beta')
        st.line_chart(self.df_factor_time.drop(columns=['Alpha','R2']))
        st.write('Alpha')
        st.line_chart(self.df_factor_time['Alpha'])
        st.write('R^2')
        st.line_chart(self.df_factor_time['R2'])
        
    def run(self):
        self.get_data()
        self.display_input()

        if st.session_state.run:
            self.pre_preprocess()
            self.fit()
            self.plot()
        
if __name__ == "__main__":
    app = SteamLit()
    app.run()