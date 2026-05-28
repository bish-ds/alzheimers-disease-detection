$(document).ready(function () {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('.gradcam-frame').hide();

    function resetResult() {
        $('#result').hide();
        $('#resultLabel').text('');
        $('#resultConfidence').text('');
        $('#resultDescription').text('');
        $('#gradcamLayer').text('');
        $('#gradcamImage').removeAttr('src');
        $('.gradcam-frame').hide();
    }

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#originalImage').attr('src', e.target.result);
                $('.image-section').fadeIn(250);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $('#imageUpload').change(function () {
        $('#btn-predict').show();
        resetResult();
        readURL(this);
    });

    $('#btn-predict').click(function () {
        var formData = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();
        resetResult();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();

                if (data.error) {
                    $('#resultLabel').text('Unable to analyze image');
                    $('#resultDescription').text(data.error);
                } else {
                    var confidence = Math.round((data.confidence || 0) * 1000) / 10;
                    $('#resultLabel').text(data.label);
                    $('#resultConfidence').text('Confidence: ' + confidence + '%');
                    $('#resultDescription').text(data.description);
                    $('#gradcamLayer').text('Grad-CAM layer: ' + data.gradcam_layer);

                    if (data.gradcam_image) {
                        $('#gradcamImage').attr('src', data.gradcam_image);
                        $('.gradcam-frame').fadeIn(250);
                    }
                }

                $('#result').fadeIn(250);
                $('#btn-predict').show();
            },
            error: function (xhr) {
                $('.loader').hide();
                var message = 'The image could not be analyzed.';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    message = xhr.responseJSON.error;
                }
                $('#resultLabel').text('Unable to analyze image');
                $('#resultDescription').text(message);
                $('#result').fadeIn(250);
                $('#btn-predict').show();
            }
        });
    });
});
