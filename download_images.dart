import 'dart:async';
import 'dart:io';

void main() async {
  final String mineralLabel = "pyrite";

  final File csvFile = File("selected_minerals/${mineralLabel}.csv");

  String csvfileAsString = await csvFile.readAsString().whenComplete(() {
    print("successfully read csv file");
  });

  final List<String> csvList = csvfileAsString.split("\n");

  final List<String> urls = [];

  for (String url in csvList) {
    List<String> tempList = url.split(',');

    urls.add(tempList.first);
    // for (String str in tempList) {
    //   if (str.contains("http")) {
    //     urls.add(str);
    //   }
    // }
  }
  print("gotten list of urls only");
  print(urls);
  await downloadImages(urls: urls, mineralLabel: mineralLabel);
}

Future<void>? downloadImages({
  required List<String> urls,
  required String mineralLabel,
}) async {
  String _basePath = "downloaded_images/";

  for (int i =0; i < urls.length-1; i++) {
    Uri uri = Uri.parse(urls[i]);

    final HttpClient client = HttpClient();
    HttpClientRequest request = await client.getUrl(uri);

    HttpClientResponse response = await request.close();
    if (response.statusCode == 200) {
      print("succesffuly downloaded image");

      response.pipe(File("$_basePath${i}_$mineralLabel.jpg").openWrite());
    } else {
      print("an error occured while getting data");
    }
  }
}
