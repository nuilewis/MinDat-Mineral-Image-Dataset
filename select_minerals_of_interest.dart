import 'dart:io';

void main() async {
  final List<String> mineralList = [
    "quartz",
    "calcite",
    "bauxite",
    "copper",
    "feldspar",
    "silver",
    "gold",
    "gneiss",
    "granite",
    "basalt",
    "fluorite",
    "pyrite",
    "magnetite",
    "galena",
    "sphalerite",
  ];

  final File csvFile = File("img_url_list_converted.csv");

  //final File csvFile = File("img_list_test.csv");
  final String csvfileAsString = await csvFile.readAsString().whenComplete(
    () {
      print("imported csv file");
    },
  );

  ///Optimize the CSV File and remove entries with multiple minerals

  List<String> cleanedMineralList = cleanDataSet(csvfileAsString);

  await File("selected_minerals/cleaned_mineral_list.csv")
      .writeAsString(cleanedMineralList.join());

  ///Select Minerals of Interest

  List<String> selectedMinerals = selectMineralOfInterest(
      InterestedMineralsList: mineralList, csv: cleanedMineralList);

  ///Writing the output file to disk

  await File("selected_minerals/selected_minerals.csv")
      .writeAsString(selectedMinerals.join());

  ///Saving each mineral as seperate file
  await saveMineralsasIndividualList(
      mineralNames: mineralList, csv: selectedMinerals);
}

//-------------------------------
List<String> selectMineralOfInterest({
  required List<String> InterestedMineralsList,
  required List<String> csv,
}) {
  //Conver the csv into a list

  List<String> listofJoinedMinerals = [];

  for (int i = 0; i < csv.length; i++) {
    //loops through and Check if the mineral is among the list of out interested minerals then add to the list

    for (String mineralName in InterestedMineralsList) {
      if (csv[i].contains(mineralName)) {
        listofJoinedMinerals
            .add(csv[i]); //Adds our mineral of interest to a new list
      }
    }
  }

  print("selected our minerals of interest");
  return listofJoinedMinerals;
}

//--------------------------------------------------------
List<String> cleanDataSet(String csv) {
  List<String> csvList =
      csv.split('\n').toList(); // Splits the list at each new line

  List<String> chosenMinerals = [];
  for (int i = 0; i < csvList.length; i++) {
    String entry = csvList[i];
    entry.trim();

    ///remove any extra space at the endg that makes it think there is a
    /// new column this include the trailing comma, a space, and a "new line" charcter
    if (entry.length > 3) {
      entry = entry.substring(
          0,
          entry.length -
              3); //Remove the comma, space, and faux space at the end
    }

    if (entry.split(' ').length == 2) {
      ///Take an entry, split it into the different words. If it has less than 2 items in the list,
      ///then its an image with only a single mineral type and lable, so it should be used.
      chosenMinerals.add('${entry}\n');
    }
  }

  print("cleaned up our list");

  return chosenMinerals;
}

Future<void> saveMineralsasIndividualList({
  required List<String> mineralNames,
  required List<String> csv,
}) async {
  for (String mineral in mineralNames) {
    List<String> tempMineral = [];
    for (int i = 0; i < csv.length; i++) {
      if (csv[i].contains(mineral)) {
        tempMineral.add(csv[i]);
      }
    }

    //Save the file

    await File("selected_minerals/${mineral}.csv")
        .writeAsString(tempMineral.join());

    tempMineral.clear();
  }

  print("saving each mineral to file");
}
