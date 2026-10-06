import streamlit as st

st.set_page_config(
    page_title="Morrow",
    page_icon="🌙"
)

st.title("🌙 Morrow")

st.write("Morrow UI is running.")

name = st.text_input("Test input")

if st.button("Test Button"):
    st.success(f"Button works! You entered: {name}")