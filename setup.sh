mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"nasskall@unipi.gr\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamli~/.streamlit/config.toml