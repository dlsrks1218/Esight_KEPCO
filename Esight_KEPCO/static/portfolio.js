/* Header */
window.onload = function(){scrollFunction()};
window.onscroll = function(){scrollFunction()};
window.onload = function(){datepicker()};

function scrollFunction(){
  var header = document.getElementById('header');

  if(document.documentElement.scrollTop > 70){
    if(!header.classList.contains('navbar-fixed')){
      header.classList.add('navbar-fixed');
      document.getElementsByTagName('body')[0].style.marginTop = '70px';
      header.style.display = 'none';
      setTimeout(function(){
        header.style.display = 'block';
      }, 40);
    }
  } else{
    if(header.classList.contains('navbar-fixed')){
      header.classList.remove('navbar-fixed');
      document.getElementsByTagName('body')[0].style.marginTop = '0';
    }
  }
}

/*toggle : 있으면 제거, 없으면 생성*/
function menuToggle(){
  document.getElementById('menu').classList.toggle('show');
}

document.getElementById('toggleBtn').addEventListener('click', menuToggle);

/* WELCOME AREA */
var imageSlideIndex = 1;
showImageSlides(imageSlideIndex);

function imageSlideTimer(){

  plusImageSlides(1);
}

var imageTimer = setInterval(imageSlideTimer, 3000);

function plusImageSlides(n){
  clearInterval(imageTimer);
  imageTimer = setInterval(imageSlideTimer, 3000);

  showImageSlides(imageSlideIndex += n);
}

function currentImageSlide(n){
  clearInterval(imageTimer);
  imageTimer = setInterval(imageSlideTimer, 3000);

  showImageSlides(imageSlideIndex = n);
}

function showImageSlides(n){
  var i;
  var slides = document.getElementsByClassName('image-slide');
  var dots = document.getElementsByClassName('dot');
  if(n > slides.length){ imageSlideIndex = 1}
  if(n < 1){ imageSlideIndex = slides.length}
  for(i = 0; i < slides.length; i++){
    slides[i].style.display = 'none';
  }
  for(i = 0; i < dots.length; i++){
    dots[i].className = dots[i].className.replace(' active', '');
  }
  slides[imageSlideIndex - 1].style.display = 'block';
  dots[imageSlideIndex - 1].className += ' active';
}

document.getElementById('imagePrev').addEventListener('click', plusImageSlides.bind(null, -1));
document.getElementById('imageNext').addEventListener('click', plusImageSlides.bind(null, 1));

document.getElementById('firstDot').addEventListener('click', currentImageSlide.bind(null, 1));
document.getElementById('secondDot').addEventListener('click', currentImageSlide.bind(null, 2));
document.getElementById('thirdDot').addEventListener('click', currentImageSlide.bind(null, 3));
document.getElementById('forthDot').addEventListener('click', currentImageSlide.bind(null, 4));

/* ABOUT AREA */
function datepicker() {

    var _inputs = document.getElementsByTagName('input');

    for (var i = 0; i < _inputs.length; i++) {
        // _inputs[i].parentNode.getElementsByClassName('result')[0].innerHTML = _inputs[i].value;
        // console.log(i)
        // console.log(_inputs[i])
        _inputs[i].onchange = function () {
            console.log(this.value);
            console.log(this.name);
            var input_data = this.value + this.name;
            // var result_node = this.parentNode.getElementsByClassName('result');
            // result_node[0].innerHTML = this.value;
            if (this.name == "1" || this.name == "2" || this.name == "4"){
                var input_data = "0" + input_data;
                $.ajax({
                    type: 'POST',
                    url: "/background_process_test",
                    data: input_data, //passing some input here
                    dataType: "text",
                    success: function(response) {
                        outputnum = "#output" + input_data.slice(-1);
                        console.log("==================");
                        console.log(outputnum);
                        console.log(response);
                        jQuery(outputnum).html(response);
                    }
                }).done(function(data){
                        console.log(data);

                });
            }
            else{
                localStorage.setItem("date", input_data);
            }
          };
    }
}

/* PORTFOLIO AREA */
function filterSelection(id){
  var date;

  date = localStorage.getItem("date");

  input_data = "1" + date + "3" + id;

  console.log(input_data)

  x = document.getElementsByClassName('listItem');


  for(i=0; i<x.length;i++){
    removeClass(x[i], 'active');
  }
  addClass(document.getElementById(id), 'active');

  $.ajax({
      type: 'POST',
      url: "/background_process_test",
      data: input_data, //passing some input here
      dataType: "text",
      success: function (response) {
          outputnum = "#output3";
          console.log("==================");
          console.log(outputnum);
          console.log(response);
          jQuery(outputnum).html(response);
      }
  }).done(function (data) {
      console.log(data);
  });

}

function filterSelection2(id){
  var date;


  input_data = "1" + "2011-05-24" + "5" + id;

  console.log(input_data)

  x = document.getElementsByClassName('listItem');


  for(i=0; i<x.length;i++){
    removeClass(x[i], 'active');
  }
  addClass(document.getElementById(id), 'active');


  $.ajax({
      type: 'POST',
      url: "/background_process_test",
      data: input_data, //passing some input here
      dataType: "text",
      success: function (response) {
          outputnum = "#output5";
          console.log("==================");
          console.log(outputnum);
          console.log(response);
          jQuery(outputnum).html(response);
      }
  }).done(function (data) {
      console.log(data);
  });

}

function addClass(element, name){
  if(element.className.indexOf(name) == -1){
    element.className += " " + name;
  }
}

function removeClass(element, name){
  var arr;
  arr = element.className.split(" ");

  while (arr.indexOf(name) > -1) {
    arr.splice(arr.indexOf(name), 1);
  }

  element.className = arr.join(" ");
}

document.getElementById('fridge').addEventListener('click', filterSelection.bind(null, 'fridge'));
document.getElementById('light').addEventListener('click', filterSelection.bind(null, 'light'));
document.getElementById('microwave').addEventListener('click', filterSelection.bind(null, 'microwave'));
document.getElementById('electric oven').addEventListener('click', filterSelection.bind(null, 'electric oven'));
document.getElementById('washer dryer').addEventListener('click', filterSelection.bind(null, 'washer dryer'));
document.getElementById('dish washer').addEventListener('click', filterSelection.bind(null, 'dish washer'));
document.getElementById('electric space heater').addEventListener('click', filterSelection.bind(null, 'electric space heater'));

document.getElementById('fridge2').addEventListener('click', filterSelection2.bind(null, 'fridge2'));
document.getElementById('light2').addEventListener('click', filterSelection2.bind(null, 'light2'));
document.getElementById('microwave2').addEventListener('click', filterSelection2.bind(null, 'microwave2'));
document.getElementById('electric oven2').addEventListener('click', filterSelection2.bind(null, 'electric oven2'));
document.getElementById('washer dryer2').addEventListener('click', filterSelection2.bind(null, 'washer dryer2'));
document.getElementById('dish washer2').addEventListener('click', filterSelection2.bind(null, 'dish washer2'));
document.getElementById('electric space heater2').addEventListener('click', filterSelection2.bind(null, 'electric space heater2'));

/* NAVBAR ANCHOR */
function moveTo(id){
  if(id == 'brand'){
    window.scrollTo(0, 0);
  }else{
    window.scrollTo(0, document.getElementById(id).offsetTop-70);
  }

  document.getElementById('menu').classList.remove('show');
}

document.getElementById('navbarBrand').addEventListener('click', moveTo.bind(null,'brand'));
document.getElementById('navbarAbout').addEventListener('click', moveTo.bind(null,'about'));
document.getElementById('navbarService').addEventListener('click', moveTo.bind(null,'service'));
document.getElementById('navbarPortfolio').addEventListener('click', moveTo.bind(null,'portfolio'));
document.getElementById('navbarPrediction').addEventListener('click', moveTo.bind(null,'prediction'));
document.getElementById('navbarDay').addEventListener('click', moveTo.bind(null,'day'));

