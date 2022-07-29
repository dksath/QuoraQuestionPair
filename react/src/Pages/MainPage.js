import {
  Container,
  VStack,
  FormControl,
  FormLabel,
  Input,
  useToast,
  Button,
  Box,
  Stack,
  Heading,
  CircularProgress, 
  CircularProgressLabel,
  Center,
} from "@chakra-ui/react";
import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Background from "../Background.jpg";

const MainPage = () => {
  const [question1, setQuestion1] = useState('');
  const [question2, setQuestion2] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();
  const navigate = useNavigate();

  function submitNav() {
    navigate("../dash", { replace: true });
  }

  // Validation of Question Page Fields
  const submitHandler = async () => {
    if (!question1 || !question2) {
      toast({
        title: "Please fill in all the fields",
        status: "warning",
        duration: 5000,
        isClosable: true,
        position: "bottom",
      });
      return;
    }

    setIsLoading(true);

    try {
      const config = {
        headers: {
          "Content-type": "application/json",
        },
      };

      // POST Input Data
      // Change this endpoint
      const { data } = await axios.post(
        "http://127.0.0.1:8000/predict",
        { sentence1: question1, 
          sentence2: question2 },
        config
      );

      toast({
        title: "Submitted Successfully",
        status: "success",
        duration: 5000,
        isClosable: true,
        position: "bottom",
      });
      submitNav();
      setIsLoading(false);
    } catch (error) {
      toast({
        title: "Error Occurred!",
        description: error.response.data.message,
        status: "error",
        duration: 5000,
        isClosable: true,
        position: "bottom",
      });
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div
      style={{
        backgroundImage: `url(${Background})`,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        backgroundSize: "cover",
        width: "100vw",
        height: "100vh",
      }}
    >
      <Center>
<Box bg="pink.100"
          w="20%"
          p={4}
          borderRadius="lg"
          borderWidth="1px"
          alignItems="center"
          m="400px 0 15px 0">
          
  <Center>
    <CircularProgress isIndeterminate color='red.300' />
      <Heading fontSize="3xl">   Sending Data...</Heading>  
    </Center>
 </Box>
    </Center>
    </div>
    )
  }

  return (
    <div
      style={{
        backgroundImage: `url(${Background})`,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        backgroundSize: "cover",
        width: "100vw",
        height: "100vh",
      }}
    >
      <Container maxW="xl" centerContent>
        <Box p={3} w="100%" m="100px 0 15px 0"></Box>
        <Box
          bg="pink.100"
          w="100%"
          p={4}
          borderRadius="lg"
          borderWidth="1px"
          m="40px 0 15px 0"
          alignItems="center"
        >
          <Heading fontSize="3xl">Welcome to Quora Question Pairs!</Heading>
        </Box>
        <Box bg="pink.100" w="100%" p={4} borderRadius="lg" borderWidth="1px">
          <VStack spacing="5px">
            <FormControl id="question1" isRequired>
              <FormLabel>Question 1</FormLabel>
              <Input
                placeholder="Enter Question 1"
                onChange={(e) => setQuestion1(e.target.value)}
                bgColor="white"
              />
            </FormControl>

            <FormControl id="email" isRequired>
              <FormLabel>Question 2</FormLabel>
              <Input
                placeholder="Enter Question 2"
                onChange={(e) => setQuestion2(e.target.value)}
                bgColor="white"
              />
            </FormControl>

            <Box p={3} w="100%" m="200px 0 15px 0"></Box>
            <Heading fontSize="2xl">We are using the BERT Model to predict the similarity</Heading>

            <Button
              colorScheme="pink"
              width="100%"
              style={{ marginTop: 15 }}
              onClick={submitHandler}
            >
              Submit
            </Button>
          </VStack>
        </Box>
      </Container>
    </div>
  );
};

export default MainPage;
