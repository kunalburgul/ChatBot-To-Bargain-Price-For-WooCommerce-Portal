$(document).ready(function() {

  $('.color-choose input').on('click', function() {
      var headphonesColor = $(this).attr('data-image');

      $('.active').removeClass('active');
      $('.left-column img[data-image = ' + headphonesColor + ']').addClass('active');
      $(this).addClass('active');
  });
});

// Check if local storage has previously stored value
// If there are values then populate relevent field with value.
document.addEventListener("DOMContentLoaded", function(event) {
var getStoredProductID = localStorage.ProductID;
var getStoredCustomerID = localStorage.CustomerID;
var getStoredQuantity = localStorage.Quantity;
console.log(getStoredProductID ,getStoredCustomerID,getStoredQuantity);
      if(getStoredProductID !==undefined || getStoredCustomerID !== undefined || getStoredQuantity !== undefined){
      document.getElementById("ProductID").value=getStoredProductID;
      document.getElementById("CustomerID").value=getStoredCustomerID;
      document.getElementById("Quantity").value=getStoredQuantity;
      }

})

// Set local storage on click of next button
var getNextButton = document.getElementById("bargain"); getNextButton.addEventListener("click",function(){

localStorage.name=document.getElementById("ProductID").value;
localStorage.age=document.getElementById("CustomerID").value;
localStorage.age=document.getElementById("Quantity").value;
})
