import 'dart:io';

void main() async {
  final List<String> mineralList = [
    "quartz",
    "calcite",
  ];

  //final File csvFile = File("img_url_list_converted.csv");

  final File csvFile = File("img_list_test.csv");
  final String csvfileAsString = await csvFile.readAsString().whenComplete(
    () {
      print("imported csv file");
    },
  );

  List<String> selectedMinerals = selectMineralOfInterest(
      InterestedMineralsList: mineralList, csv: csvfileAsString);

  ///Writing the file to disk

  File outPutCSV = File("selected_minerals/selected_minerals.csv");

  await outPutCSV.writeAsString(selectedMinerals.join());
}

List<String> selectMineralOfInterest({
  required List<String> InterestedMineralsList,
  required String csv,
}) {
  //Conver the csv into a list

  List<String> listofJoinedMinerals = [];

  List<String> splitList =
      csv.split('\n').toList(); // Splits th list at each new line

  for (int i = 0; i < splitList.length; i++) {
    //loops through and Check if the mineral is among the list of out interested minerals then add to the list

    for (String mineralName in InterestedMineralsList) {
      if (splitList[i].contains(mineralName)) {
        listofJoinedMinerals
            .add(splitList[i]); //Adds our mineral of interest to a new list
      }
    }
  }
  print("final list with minerals of interest");
  print(listofJoinedMinerals.toList());

  //Write the list to a csv file
  return listofJoinedMinerals;
}
