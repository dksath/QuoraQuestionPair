import {
  Box,
  Table,
  TableContainer,
  Thead,
  Tr,
  Td,
  Th,
  Tbody,
} from "@chakra-ui/react";
import React from "react";

const UserInputs = ({ question1, question2 }) => {
  return (
    <Box
      d="flex"
      justifyContent="center"
      p={3}
      bg="blue.50"
      w="100%"
      m="20px 0 15px 0"
      borderRadius="lg"
      borderWidth="1px"
      boxShadow="md"
    >
      <TableContainer>
        <Table variant="striped" colorScheme="blue" size="sm">
          <Thead>
            <Tr>
              <Th>Index</Th>
              <Th>Questions</Th>
            </Tr>
          </Thead>
          <Tbody>
            <Tr>
              <Td>1</Td>
              <Td>{question1}</Td>
            </Tr>
            <Tr>
              <Td>2</Td>
              <Td>{question2}</Td>
            </Tr>
          </Tbody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default UserInputs;
