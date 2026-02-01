import streamlit as st
st.title('Homework Manager')
HW1 = st.Page('HW/HW1.py', title = 'HW1', icon = '💻')
HW2 = st.Page('HW/HW2.py', title = 'HW2', icon = '💻')
pg = st.navigation([HW1, HW2])
st.set_page_config(page_title = 'Homework Manager',
                   initial_sidebar_state='expanded')
pg.run()
