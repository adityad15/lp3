// SPDX-License-Identifier: UNLICENSED

pragma solidity >= 0.7.0 <0.9.0;

// Build the Contract
contract MarksManagmtSys
{
    // Create a structure for student details
    struct StudentStruct
    {
        uint ID;
        string fName;
        string lName;
        uint marks;
    }

    address owner;

    uint public stdCount = 0;

    // Create Array to store Student data
    StudentStruct[] stdRecords;

    constructor()
    {
        owner = msg.sender;
    }

    // Create a function to add the new records
    function addNewRecords(uint _ID,
                           string memory _fName,
                           string memory _lName,
                           uint _marks) public payable
    {
        // Increase the count by 1
        stdCount = stdCount + 1;

        // Adding data into array
        stdRecords.push(StudentStruct(_ID, _fName, _lName, _marks));
    }

    function getAllRecords() public view returns(StudentStruct[] memory)
    {
        return stdRecords;
    }
}
