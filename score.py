import urllib.request

# Load image data
with open("image.jpg", "rb") as image_file:
    body = image_file.read()

url = 'http://e9cddab6-15c5-4a4d-bdee-6101f526ee75.eastus.azurecontainer.io/score'

# Update headers for image data
headers = {'Content-Type': 'application/octet-stream', 'Accept': 'application/json'}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
