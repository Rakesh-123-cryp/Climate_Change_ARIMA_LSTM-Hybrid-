from flask import Flask, render_template

app = Flask(__name__)
app.secret_key = "abcd"



#Home Page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about_this")
def about_this():
    return render_template("about_this.html")

@app.route("/about_us")
def about_us():
    return render_template("/about_us.html")

@app.route("/aqi")
def aqi():
    return render_template("/aqi.html")

@app.route("/heatwaves_graph")
def heatwaves_graph():
    return render_template("/heatwaves_graph.html")

@app.route("/heatwaves_tables")
def heatwaves_tables():
    return render_template("/heatwaves_tables.html")

@app.route("/aqi_graph")
def aqi_graph():
    return render_template("/aqi_graph.html")

@app.route("/aqi_tables")
def aqi_tables():
    return render_template("/aqi_tables.html")


@app.errorhandler(404)
def error_page(e):
    return render_template("error_page.html")
    

if(__name__ == "__main__"):
    app.run(host='0.0.0.0', port='8080', debug = True)