import {
  Container,
  Button,
  Box,
  Heading,
  Center,
  CircularProgress,
} from "@chakra-ui/react";
import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Background from "../Background.jpg";
import UserInputs from "../Components/UserInputs";

const Dashboard = () => {
  const [getData, setGetData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  function submitNav() {
    navigate("..", { replace: true });
  }

  // GET Model Results without Polling
    useEffect(() => {
    const callAxios = async () => {
      await axios
        .get("http://127.0.0.1:8000/result")
        .then(response => {
          const items = response.data.items;
          setGetData(items);
          console.log('SUCCESS', response);
          setIsLoading(false);
        })
        .catch(error => {
          console.log(error);
        });
    };
    callAxios();
  }, []);

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
      <Heading fontSize="3xl">   Calculating...</Heading>
   
    </Center>
 </Box>
    </Center>
    </div>
    )
  }

  if (getData.length > 0) {
    return getData.map((item, index) => {
      console.log(item);
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
              <Heading fontSize="3xl">Model Results</Heading>

              <UserInputs
                question1={item.sentence1}
                question2={item.sentence2}
              />

              <Heading fontSize="xl">These questions are {item.value}</Heading>

              <Button
                colorScheme="pink"
                width="100%"
                style={{ marginTop: 15 }}
                onClick={submitNav}
              >
                Try Other Question Pairs
              </Button>
            </Box>
          </Container>
        </div>
      );
    });
  }
};

export default Dashboard;
