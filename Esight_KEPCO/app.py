from flask import Flask, request, render_template, jsonify
import visualization as vis

app = Flask(__name__)
app.secret_key = 'SHH!'


@app.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    #     print(request.get_data())
    # else:
    #     message = {'gretting' : 'hello from flask'}
    #     return jsonify(message)
    return render_template('Esight.html')


@app.route("/background_process_test", methods=['POST'])
def background_process_test():
    selected_item = request.get_data().decode('ascii')
    date = selected_item[1:11]
    num = selected_item[11]
    print(date, num)
    print(selected_item)
    graph = ""
    if (selected_item[0] == "0"):
        print("시간별, 지역별 통계 데이터 요청")
        if num == "1":
            graph = vis.vis1(date,date,6)
        elif num == "2":
            graph = vis.vis2(2,date,date)
        elif num == "4":
            print("num 4 selected")
            graph = vis.vis4(date,date,6)
        else:
            print("wrong access")
            pass
    elif (selected_item[0] == "1"):
        if num == "3":
            print("가전제품 소비 패턴 통계 데이터 요청")
            appliance_info = selected_item[13:]
            print("========")
            print(appliance_info)
            print("======")
            graph = vis.vis3(date, date, appliance_info)
        elif num == "5":
            print("가전제품 소비 패턴 예측 요청")
            appliance_info = selected_item[12:]
            print(appliance_info)
            graph = vis.vis5(appliance_info)
    else:
        print("wrong input_value")
        pass

    print(graph)

    return graph


if __name__ == '__main__':
    app.run(host='0.0.0.0')