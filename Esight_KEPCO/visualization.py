from nilmtk_func import *


# applinacesTimeUsage
def vis1(start, end, buildings_total):
    total = total_usage_per_hour(buildings_total, False).loc[start:end]
    timeData = total.index.strftime('%H').tolist()
    appliances = ["fridge", "light", "microwave", "electric oven", "washer dryer", "dish washer",
                  "electric space heater"]
    data = {'timestamp': timeData}
    for a in appliances:
        tmp = applianceTimeUsage(a, buildings_total, start, end)
        tmp = tmp.loc[start:end]
        tmp = (tmp['active'] / total['active']).values
        data[a] = np.nan_to_num(tmp, copy=False).tolist()
    tmpTotal = [0] * len(timeData)
    for appliance in appliances:
        for l in range(len(timeData)):
            tmpTotal[l] += data[appliance][l]
    for appliance in appliances:
        for l in range(len(timeData)):
            data[appliance][l] = round((data[appliance][l] / tmpTotal[l]) * 100, 2)

    colors = Pastel2[len(appliances)]
    p = figure(x_range=data['timestamp'], plot_height=500, plot_width=1166, title="시간별 평균 활동(전국)",
               toolbar_location=None, tools="hover", tooltips="$name @timestamp: @$name")
    p.vbar_stack(appliances, x='timestamp', width=0.5, color=colors, source=data,
                 legend=[value(x) for x in appliances])
    p.y_range.start = 0
    p.y_range.end = 100
    p.x_range.range_padding = 0.5
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "vertical"
    p.legend.click_policy = "hide"
    script_bok, div_bok = components(p)

    return render_template('update_content.html', div_bok=div_bok, script_bok=script_bok)

def vis4(start, end, buildings_total):
    total = total_usage_per_hour(buildings_total, False).loc[start:end]
    timeData = total.index.strftime('%H').tolist()
    appliances = ["fridge", "light", "microwave", "electric oven", "washer dryer", "dish washer",
                  "electric space heater"]
    data = {'timestamp': timeData}
    for a in appliances:
        tmp = applianceTimeUsage(a, buildings_total, start, end)
        tmp = tmp.loc[start:end]
        tmp = (tmp['active'] / total['active']).values
        data[a] = np.nan_to_num(tmp, copy=False).tolist()
    tmpTotal = [0] * len(timeData)
    for appliance in appliances:
        for l in range(len(timeData)):
            tmpTotal[l] += data[appliance][l]
    for appliance in appliances:
        for l in range(len(timeData)):
            data[appliance][l] = round((data[appliance][l] / tmpTotal[l]) * 100, 2)

    import copy
    x = copy.deepcopy(data)
    del x['timestamp']
    for k in x.keys():
        tmp = x[k]
        x[k] = sum(tmp, 0.0) / len(tmp)
    data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    data['color'] = Category20c[len(x)]

    p = figure(plot_height=800, plot_width=1166, title="일일 평균 활동", toolbar_location=None,
               tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='country', source=data)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    script_bok, div_bok = components(p)

    return render_template('update_content.html', div_bok=div_bok, script_bok=script_bok)

def vis2(buildingNum, start, end):
    # data
    print(start, end)
    t2 = applianceLocationUsage(buildingNum, start, end)

    print(t2)

    p1 = figure(x_range=t2['timestamp'], plot_height=500, plot_width=1166, title="지역별 평균 활동",
                toolbar_location=None, tools="hover", tooltips="$name @timestamp: @$name")

    legendKeyTmp = t2.keys()
    legendTmp = []
    for k in legendKeyTmp:
        if k == 'timestamp': continue
        for i in range(len(t2[k])):
            t2[k][i] = round(t2[k][i] * 100, 2)
        legendTmp.append(k)
    if len(legendTmp) > 20:
        colors = Category20c[20][:len(legendTmp)]
    else:
        colors = Category20c[len(legendTmp)]

    print(len(legendTmp))
    print(len(colors))

    p1.vbar_stack(legendTmp, x='timestamp', width=0.3, color=colors, source=t2,
                  legend=[value(x) for x in legendTmp])

    p1.y_range.start = 0
    p1.y_range.end = 100
    p1.x_range.range_padding = 0.5
    p1.xgrid.grid_line_color = None
    p1.axis.minor_tick_line_color = None
    p1.outline_line_color = None
    p1.legend.location = "top_left"
    p1.legend.orientation = "vertical"
    p1.legend.click_policy = "hide"

    script_bok, div_bok = components(p1)

    return render_template('update_content.html', div_bok=div_bok, script_bok=script_bok)


def vis3(start, end, appliance):
    tmp = applianceTimeUsage(appliance, buildings_count, start, end)
    tmp = tmp.loc[start:end]

    total = total_usage_per_hour(buildings_count, False).loc[start:end]

    tmp = (tmp['active'] / total['active']).values
    result = np.nan_to_num(tmp, copy=False).tolist()
    result = [round(i * 100, 2) for i in result]
    fruits = total.index.strftime('%H').tolist()
    years = [appliance]
    colors = ["#718dbf"]
    data = {'fruits': fruits,
            appliance: result}

    print(fruits, result)
    print(len(fruits), len(result))

    p = figure(x_range=fruits, plot_height=500, plot_width=1166, title=appliance + " 소비 패턴",
               toolbar_location=None, tools="hover", tooltips="$name @fruits: @$name")

    p1 = p.vbar_stack(years, x='fruits', width=0.9, color=colors, source=data,
                      legend=[value(x) for x in years])
    p.y_range.start = 0
    p.x_range.range_padding = 0.5
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    script_bok, div_bok = components(p)

    return render_template('update_content.html', div_bok=div_bok, script_bok=script_bok)

def laprnd(loc, scale):
    from scipy.stats import laplace
    from sympy import Symbol, exp, sqrt, pi, Integral
    import math
    import matplotlib.pyplot as plt
    s = laplace.rvs(loc, scale, None)
    return s

def vis5(appliance_info):
    # from scipy.stats import laplace
    # from sympy import Symbol, exp, sqrt, pi, Integral
    # import math
    import matplotlib.pyplot as plt
    # minute_predict
    # minute_cutted

    appliance_info = appliance_info[:-1]
    result_predict_df = pd.read_csv('data/minute_predict_.csv', index_col='timestamp', parse_dates=True)

    del result_predict_df['index']
    # del result_predict_df['FALSE'] #minute_predict 일 경우 데이터에 FALSE가 없어서 del할 필요없음

    for i, v in enumerate(result_predict_df[appliance_info]):
        result_predict_df[appliance_info][i] = v + laprnd(0, 0.031)

    new_graph_name = "graph" + appliance_info + ".png"

    # result_predict_df[appliance_info].resample(rule='H').mean().plot()
    plt.clf()
    plt.plot(result_predict_df[appliance_info])
    plt.xticks(rotation=20)
    # plt.savefig('static/' + new_graph_name)
    plt.savefig('static/img/' + new_graph_name)

    return render_template('outputs.html' , graph = new_graph_name)

    # return render_template('update_content.html', div_bok=div_bok, script_bok=script_bok)