function convertData(dataList)
    {var temp2 = dataList.innerText;
    temp2 = temp2.replace(/'/g,'')
    temp2 = temp2.replace(/' '/g,'')
    temp2 = temp2.split(',');
    temp2[0]= temp2[0].replace('[','') 
    temp2[temp2.length-1]= temp2[temp2.length-1].replace('] ','')
    temp2[temp2.length-1]= temp2[temp2.length-1].replace(']','')
    return temp2;}

function getIndicesOf(searchStr, str, caseSensitive) {
    var searchStrLen = searchStr.length;
    if (searchStrLen == 0) {
        return [];
    }
    var startIndex = 0, index, indices = [];
    if (!caseSensitive) {
        str = str.toLowerCase();
        searchStr = searchStr.toLowerCase();
    }
    while ((index = str.indexOf(searchStr, startIndex)) > -1) {
        indices.push(index);
        startIndex = index + searchStrLen;
    }
    return indices;
}

function strTo2dp(StrNum){
    var dp2Str = Number(StrNum);
    dp2Str =parseFloat(Math.round(dp2Str * 100) / 100).toFixed(2);
    return dp2Str;}


function inFormula(featuresChk,regressTerm) {
  for (var j = 0; j < featuresChk.length; j++) {
      var withSpaces = " ".concat(featuresChk[j]," ");
      var withBeginBracket =featuresChk[j].concat("(");
      var withEndBracket = featuresChk[j].concat(")");
      var withMultiply = featuresChk[j].concat("*");
      var withSquared = featuresChk[j].concat("<sup>2</sup>");
      var fullList = [withSpaces,withBeginBracket,withEndBracket,withMultiply,withSquared];
      var test2=fullList.some(el => regressTerm.includes(el));
      if (test2) {
            return true;
       }
  }
  return false;
}

var simplifiedRegressFormula = document.querySelector('#simplified-equation')
var inputSpreadData=(document.querySelector('#spreadsheet-data'))
var inputWeightData=(document.querySelector('#weights'))
var inputIntercept=(document.querySelector('#intercept'))
var inputRegressionType=(document.querySelector('#regression-type'))

inputSpreadData.style.display = "none"
inputWeightData.style.display = "none"
inputIntercept.style.display = "none"
inputRegressionType.style.display = "none"
debugger;
var featureList=document.querySelector('#features');
features= convertData(featureList);
var myFieldSet = document.getElementById("Select-features"); 
for (var i = 0; i < features.length; i++) {
        var nextFeature = features[i];
        var label= document.createElement("label");
        var checkbox = document.createElement("input");

        checkbox.type = "checkbox";    // make the element a checkbox
        checkbox.name = "chkFeatures";      // give it a name we can check on the server side
        checkbox.value = nextFeature;         // make its value "pair"

        label.appendChild(document.createTextNode(nextFeature)); 
        myFieldSet.appendChild(checkbox);   // add the box to the element
        myFieldSet.appendChild(label);// add the description to the element

}

myFieldSet = document.getElementById("Select-features"); 

var button = document.querySelector("#Select-button");

const hideBox = document.querySelector('#simplify');
hideBox.checked = false
myFieldSet.style.display = "none"
featureList.style.display = "none"
hideBox.addEventListener('change', function(){
  if(hideBox.checked){
    myFieldSet.style.display = "block";
    simplifiedRegressFormula.style.display = "block";   
  } else {
    myFieldSet.style.display = "none";
    simplifiedRegressFormula.style.display = "none"; 
      
  }
});



button.addEventListener ("click", function(e) {
var featuresChk=[];
var temp= document.querySelectorAll("input[name=chkFeatures]");
for (var i = 0; i < temp.length; i++) {  
    if (temp[i].checked == true) {    
        temp2=temp[i].value;
        temp2 = temp2.replace(' ','');
        featuresChk.push(temp2)}
}

var regressFormula = document.querySelector('#regression-formula');
regressFormula = regressFormula.innerHTML
regressFormula=regressFormula.substring(regressFormula.lastIndexOf('=') +1) 
regressFormula = regressFormula.replace(/ /,'')   
if (regressFormula[0]=="-"){
  regressFormula=regressFormula.substring(1)   
}
regressFormula=regressFormula.split('+').join(',').split('-').join(',').split(','); 
if  (isNaN(Number(regressFormula[0]))==false){
    regressFormula.shift()
}
    
var spread_data=convertData(document.querySelector('#spreadsheet-data'))
var weight_data=convertData(document.querySelector('#weights'))
var intercept=convertData(document.querySelector('#intercept'))
var newWeight=0;
var nextSpread =0;  
var nextWeight=0;
var simplified_regress= ""
for (var i = 0; i < weight_data.length; i++) {
    
    var regressTerm= regressFormula[i];    
    if  (isNaN(Number(regressTerm))) {  // checks regressTerm is not just a number
        if(inFormula(featuresChk,regressTerm)){
            if(Math.sign(Number(weight_data[i]))==-1){
                         regressTerm = "- ".concat(regressTerm);}
            else {regressTerm = "+ ".concat(regressTerm);}
            simplified_regress += regressTerm;}
        else {
            nextSpread = Number(spread_data[i]);
            nextWeight = Number(weight_data[i]);
            newWeight += nextSpread*nextWeight;}}  
}


    
newWeight +=Number(intercept);
newWeight =strTo2dp(newWeight)
if (inputRegressionType.innerHTML=="logistic"){
  simplified_regress= "<b>w</b><sup>T</sup><b>x</b> = ".concat(newWeight," ",simplified_regress); 
} else {
simplified_regress= "prediction = ".concat(newWeight," ",simplified_regress)};    
simplifiedRegressFormula.innerHTML=simplified_regress;
e.preventDefault();
});