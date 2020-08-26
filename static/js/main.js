console.log('test')

$(document).ready(function() {
  $('.submit').click(function(event) {
    console.log('Clicked button')

    var fname = $('.fname').val()
    var sname = $('.sname').val()
    var email = $('.email').val()
    var message = $('.message').val()
    var statusElm = $('.status')
    statusElm.empty()

      if(fname.length > 2 && sname.length > 2) {
        console.log('Names are valid')
      } else {
        statusElm.append('<div>Name(s) are invalid</div>')
        event.preventDefault()
      }

      if(email.length > 6 && email.includes('@') && email.includes('.')) {
        console.log('Message is valid')
      } else {
        statusElm.append('<div>Email is invalid</div>')
        event.preventDefault()
      }

      if(message.length > 15) {
        console.log('Message is valid')
      } else {
        statusElm.append('<div>Message is invalid - minimum 15 characters required</div>')
        event.preventDefault()
      }
      if(fname.length > 2 && sname.length > 2 && email.length > 6 && email.includes('@') && email.includes('.') && message.length > 15) {
        statusElm.append('<div>Form submitted :)</div>')
      }
  })
})
